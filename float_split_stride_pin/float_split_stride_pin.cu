#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>          // at::BFloat16 helpers
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>    // CUDAStreamGuard / CUDAGuard

//----------------------------------------------------------------------------
//  utilities: indexer <sizes,strides,offset>  →  flatten / offset
//----------------------------------------------------------------------------
template<int N>
struct StridedIndex {
    int64_t sizes[N];
    int64_t strides[N];
    int64_t base_offset;                        // elements, not bytes
    bool    is_contig = false;  // ✅ 添加标志

    __host__ __device__ __forceinline__
    int64_t offset(int64_t linear) const {
        if (is_contig) {
            return base_offset + linear;
        }
        int64_t off = base_offset;
        #pragma unroll
        for (int i = N - 1; i >= 0; --i) {
            int64_t cur = linear % sizes[i];
            linear     /= sizes[i];
            off        += cur * strides[i];
        }
        return off;
    }
};

template<int N>
StridedIndex<N> make_indexer(const at::Tensor& t) {
    StridedIndex<N> ix;
    auto sz = t.sizes();
    auto st = t.strides();
    for (int i = 0; i < N; ++i) {
        ix.sizes[i]   = (i < sz.size()) ? sz[i] : 1;
        ix.strides[i] = (i < st.size()) ? st[i] : 1;
    }
    ix.base_offset = 0;
    ix.is_contig = t.is_contiguous();
    return ix;
}

//----------------------------------------------------------------------------
//  kernels (<<<grid,256>>>),  N=4 covers up to 4-D activations
//----------------------------------------------------------------------------
template<int N>
__global__ void pack_bf16_kernel(const uint16_t* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint16_t bits = __ldg(&in[ix.offset(idx)]);
    exp_out[idx]  = (bits >> 7) & 0xFF;
    sm_out[idx]   = ((bits >> 15) & 0x1) << 7 | (bits & 0x7F);
}

template<int N>
__global__ void pack_bf16_kernel_vec2(const uint16_t* __restrict__ in,
                                      StridedIndex<N> ix,
                                      uint8_t* __restrict__ exp_out,
                                      uint8_t* __restrict__ sm_out,
                                      int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 >= numel) return;

    int64_t off = ix.offset(idx);
    uint16_t bits0 = __ldg(&in[ off ]);
    uint16_t bits1 = __ldg(&in[ off + 1 ]);

    exp_out[idx]     = (bits0 >> 7) & 0xFF;
    sm_out[idx]      = ((bits0 >> 15) & 0x1) << 7 | (bits0 & 0x7F);
    exp_out[idx + 1] = (bits1 >> 7) & 0xFF;
    sm_out[idx + 1]  = ((bits1 >> 15) & 0x1) << 7 | (bits1 & 0x7F);
}

template<int N>
__global__ void pack_fp32_kernel(const float* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint32_t bits = __ldg(& reinterpret_cast<const uint32_t*>(in)[ ix.offset(idx) ]);
    exp_out[idx]  = (bits >> 23) & 0xFF;

    uint32_t sm24 = ((bits & 0x7FFFFF)      )      // mant
                  | ((bits >> 8) & 0x800000);      // sign
    sm_out[idx*3+0] =  sm24        & 0xFF;
    sm_out[idx*3+1] = (sm24 >> 8 ) & 0xFF;
    sm_out[idx*3+2] = (sm24 >> 16) & 0xFF;
}

template<int N>
__global__ void pack_fp32_kernel_vec2(const float* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 >= numel) return;

    int64_t off = ix.offset(idx);
    uint32_t bits = __ldg(& reinterpret_cast<const uint32_t*>(in)[ off ]);
    uint32_t bits_1 = __ldg(& reinterpret_cast<const uint32_t*>(in)[ off + 1 ]);
    exp_out[idx]  = (bits >> 23) & 0xFF;
    exp_out[idx+1]  = (bits_1 >> 23) & 0xFF;

    uint32_t sm24 = ((bits & 0x7FFFFF)      )      // mant
                  | ((bits >> 8) & 0x800000);      // sign
    uint32_t sm24_1 = ((bits_1 & 0x7FFFFF)      )      // mant
                  | ((bits_1 >> 8) & 0x800000);      // sign

    sm_out[idx*3+0] =  sm24        & 0xFF;
    sm_out[idx*3+1] = (sm24 >> 8 ) & 0xFF;
    sm_out[idx*3+2] = (sm24 >> 16) & 0xFF;

    sm_out[(idx+1)*3+0] =  sm24_1        & 0xFF;
    sm_out[(idx+1)*3+1] = (sm24_1 >> 8 ) & 0xFF;
    sm_out[(idx+1)*3+2] = (sm24_1 >> 16) & 0xFF;

}

template<int N>
__global__ void unpack_bf16_kernel(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   uint16_t* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint8_t  sm   = __ldg(&sm_in[idx]);
    uint16_t sign = (sm >> 7) & 0x1;
    uint16_t mant =  sm & 0x7F;
    uint16_t bits = (sign << 15) |
                    (static_cast<uint16_t>(__ldg(&exp_in[idx])) << 7) |
                    mant;
    out[ix.offset(idx)] = bits;
}

template<int N>
__global__ void unpack_bf16_kernel_vec2(const uint8_t* __restrict__ exp_in,
                                        const uint8_t* __restrict__ sm_in,
                                        StridedIndex<N> ix,
                                        uint16_t* __restrict__ out,
                                        int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 >= numel) return;

    uint8_t sm0 = __ldg(&sm_in[idx]);
    uint8_t sm1 = __ldg(&sm_in[idx + 1]);
    uint8_t exp0 = __ldg(&exp_in[idx]);
    uint8_t exp1 = __ldg(&exp_in[idx + 1]);

    uint16_t bits0 = ((sm0 >> 7) << 15) | (static_cast<uint16_t>(exp0) << 7) | (sm0 & 0x7F);
    uint16_t bits1 = ((sm1 >> 7) << 15) | (static_cast<uint16_t>(exp1) << 7) | (sm1 & 0x7F);

    int64_t off = ix.offset(idx);
    out[ off ]     = bits0;
    out[ off + 1 ] = bits1;
}

template<int N>
__global__ void unpack_fp32_kernel(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   float* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    uint32_t sm24 =  __ldg(&sm_in[idx*3+0])
                   | (__ldg(&sm_in[idx*3+1]) << 8 )
                   | (__ldg(&sm_in[idx*3+2]) << 16);

    uint32_t sign = (sm24 >> 23) & 0x1;
    uint32_t mant =  sm24 & 0x7FFFFF;

    uint32_t bits = (sign << 31)
                  | (static_cast<uint32_t>(__ldg(& exp_in[idx])) << 23)
                  |  mant;
    reinterpret_cast<uint32_t*>(out)[ ix.offset(idx) ] = bits;
}

template<int N>
__global__ void unpack_fp32_kernel_vec2(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   float* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 >= numel) return;

    uint32_t sm24 =  __ldg(&sm_in[idx*3+0])
                   | (__ldg(&sm_in[idx*3+1]) << 8 )
                   | (__ldg(&sm_in[idx*3+2]) << 16);
    uint32_t sm24_1 =  __ldg(&sm_in[(idx+1)*3+0])
                   | (__ldg(&sm_in[(idx+1)*3+1]) << 8 )
                   | (__ldg(&sm_in[(idx+1)*3+2]) << 16);

    uint32_t sign = (sm24 >> 23) & 0x1;
    uint32_t sign_1 = (sm24_1 >> 23) & 0x1;
    uint32_t mant =  sm24 & 0x7FFFFF;
    uint32_t mant_1 =  sm24_1 & 0x7FFFFF;

    uint32_t bits = (sign << 31)
                  | (static_cast<uint32_t>(__ldg(& exp_in[idx])) << 23)
                  |  mant;
    uint32_t bits_1 = (sign_1 << 31)
                  | (static_cast<uint32_t>(__ldg(& exp_in[idx+1])) << 23)
                  |  mant_1;
    
    int64_t off = ix.offset(idx);
    reinterpret_cast<uint32_t*>(out)[ off ] = bits;
    reinterpret_cast<uint32_t*>(out)[ off + 1 ] = bits_1;
}
//----------------------------------------------------------------------------
//  host wrappers
//----------------------------------------------------------------------------
static inline dim3 grid_for(int64_t n) { return dim3((n + 255) / 256); }

std::vector<at::Tensor> pack_tensor(const at::Tensor& t, unsigned long long stream_ptr) {
    TORCH_CHECK(t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(t.scalar_type()==at::kFloat || t.scalar_type()==at::kBFloat16,
                "dtype must be fp32 / bf16");
    
    // cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int dev_idx = t.device().index();             // 例如 0、1…
    TORCH_CHECK(dev_idx >= 0, "sm must be on CUDA");
    cudaStream_t  raw = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream s = c10::cuda::getStreamFromExternal(raw, dev_idx);
    c10::cuda::CUDAStreamGuard guard{s};          // ⬅ 切换 thread-local current stream

    const int64_t N = t.numel();
    dim3 grid = grid_for(N);

    // auto exp = at::empty_like(t, t.options().dtype(at::kByte));       // [N] --> Wrong!
    auto exp_host = at::empty(N, at::dtype(at::kByte).pinned_memory(true));

    if (t.scalar_type() == at::kFloat) {
        auto sm  = at::empty({N,3}, t.options().dtype(at::kByte));    // [N,3]

        // ③ kernel 直接写 host pointer（使用 cudaHostGetDevicePointer）
        uint8_t* host_ptr = exp_host.data_ptr<uint8_t>();
        uint8_t* dev_ptr;
        cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);

        auto ix  = make_indexer<4>(t);

        if (ix.is_contig)
            pack_fp32_kernel_vec2<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                // exp.data_ptr<uint8_t>(),
                dev_ptr,
                sm.data_ptr<uint8_t>(), N);
        else
            pack_fp32_kernel<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                // exp.data_ptr<uint8_t>(),
                dev_ptr,
                sm.data_ptr<uint8_t>(), N);
        
        // sm.record_stream(s);
        return {exp_host, sm};

    } else {    // bf16
        // auto sm  = at::empty_like(t, at::dtype(at::kByte));           // [N] --> Wrong!
        auto sm = at::empty(N, t.options().dtype(at::kByte));
        // ③ kernel 直接写 host pointer（使用 cudaHostGetDevicePointer）
        uint8_t* host_ptr = exp_host.data_ptr<uint8_t>();
        uint8_t* dev_ptr;
        cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);

        auto ix  = make_indexer<4>(t);
        
        if (ix.is_contig) 
            pack_bf16_kernel_vec2<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                // exp.data_ptr<uint8_t>(),
                dev_ptr,                        // 写到 host
                sm.data_ptr<uint8_t>(), N);
        else
            pack_bf16_kernel<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                // exp.data_ptr<uint8_t>(),
                dev_ptr,                        // 写到 host
                sm.data_ptr<uint8_t>(), N);
        
        // sm.record_stream(s);
        return {exp_host, sm};
    }
}

// util: 计算 offset+sizes,strides 需要的底层 storage 大小
int64_t required_storage(int64_t base_off,
            const std::vector<int64_t>& sizes,
            const std::vector<int64_t>& strides) {

    int64_t max_off = base_off;

    for (size_t i = 0; i < sizes.size(); ++i)
        max_off += (sizes[i] - 1) * strides[i];

    return max_off + 1;                         // 元素个数
}


at::Tensor unpack_tensor(at::Tensor exp,
                         at::Tensor sm,
                         std::vector<int64_t> sizes,
                         std::vector<int64_t> strides,
                         int64_t storage_offset,
                         c10::ScalarType dtype, 
                         unsigned long long stream_ptr) {

    // TORCH_CHECK(exp.is_cuda() && sm.is_cuda(), "inputs must be CUDA uint8");
    TORCH_CHECK(sm.is_cuda(), "sm must be CUDA uint8");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16, "dtype mismatch");

    // cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    int dev_idx = sm.device().index();             // 例如 0、1…
    TORCH_CHECK(dev_idx >= 0, "sm must be on CUDA");
    cudaStream_t  raw = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream s = c10::cuda::getStreamFromExternal(raw, dev_idx);
    c10::cuda::CUDAStreamGuard guard{s};          // ⬅ 切换 thread-local current stream

    // -------- 0.  将 exp 指针转成 GPU 可见地址 --------
    uint8_t* exp_dev_ptr = nullptr;
    std::optional<at::Tensor> tmp_gpu_exp;         // 若需 copy

    if (exp.is_cuda()) {
        exp_dev_ptr = exp.data_ptr<uint8_t>();
        // std::cout << "exp is on cuda!" << std::endl;
    } else {
        TORCH_CHECK(exp.is_pinned(),
                    "exp on CPU must be pinned for zero-copy");

        // 尝试映射成设备指针
        cudaError_t err = cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&exp_dev_ptr),
                exp.data_ptr(), 0);

        if (err != cudaSuccess) {
            /* 某些旧 GPU 或禁用了 UVA：退回一次 H→D copy */
            tmp_gpu_exp.emplace(
                exp.to(sm.device(), /*non_blocking=*/true));
            exp_dev_ptr = tmp_gpu_exp->data_ptr<uint8_t>();
        }
    }
    
    // -------- 1.  allocate output tensor (原 stride) ----------
    const int64_t need = required_storage(storage_offset, sizes, strides);
    const int64_t N = exp.numel();
    dim3 grid = grid_for(N);
    auto flat = at::empty({need}, sm.options().dtype(dtype));   // ① 先建 1-D storage
    auto out  = flat.as_strided(sizes, strides, storage_offset); // ② 再做 view
    auto ix = make_indexer<4>(out);


    // 2) launch kernel
    if (dtype == at::kFloat) {
        if (ix.is_contig)
            unpack_fp32_kernel_vec2<4><<<grid,256,0,raw>>>(
                // exp.data_ptr<uint8_t>(),
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                out.data_ptr<float>(),
                N);
        else
            unpack_fp32_kernel<4><<<grid,256,0,raw>>>(
                // exp.data_ptr<uint8_t>(),
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                out.data_ptr<float>(),
                N);
    } else {
        if (ix.is_contig)
            unpack_bf16_kernel_vec2<4><<<grid,256,0,raw>>>(
                // exp.data_ptr<uint8_t>(),
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
                N);
        else
            unpack_bf16_kernel<4><<<grid,256,0,raw>>>(
                // exp.data_ptr<uint8_t>(),
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
                N);
    }
    out.record_stream(s);
    return out;
}

//----------------------------------------------------------------------------
//  pybind
//----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split", &pack_tensor, "pack tensor -> (exp, sm)");
    m.def("merge", &unpack_tensor,
          "unpack (exp,sm,sizes,strides,offset,dtype,stream) -> tensor",
          py::arg("exp"), py::arg("sm"),
          py::arg("sizes"), py::arg("strides"), py::arg("storage_offset"),
          py::arg("dtype"), py::arg("stream_ptr"));
}
