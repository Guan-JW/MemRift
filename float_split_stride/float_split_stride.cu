#include <torch/extension.h>
#include <cuda_bf16.h>          // at::BFloat16 helpers
#include <cstdint>

//----------------------------------------------------------------------------
//  utilities: indexer <sizes,strides,offset>  →  flatten / offset
//----------------------------------------------------------------------------
template<int N>
struct StridedIndex {
    int64_t sizes[N];
    int64_t strides[N];
    int64_t base_offset;                        // elements, not bytes

    __host__ __device__ __forceinline__
    int64_t offset(int64_t linear) const {
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
    // ix.base_offset = t.storage_offset();
    ix.base_offset = 0;
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
    uint16_t bits = in[ix.offset(idx)];
    exp_out[idx]  = (bits >> 7) & 0xFF;
    sm_out[idx]   = ((bits >> 15) & 0x1) << 7 | (bits & 0x7F);
}

template<int N>
__global__ void pack_fp32_kernel(const float* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint32_t bits = reinterpret_cast<const uint32_t*>(in)[ ix.offset(idx) ];
    exp_out[idx]  = (bits >> 23) & 0xFF;

    uint32_t sm24 = ((bits & 0x7FFFFF)      )      // mant
                  | ((bits >> 8) & 0x800000);      // sign
    sm_out[idx*3+0] =  sm24        & 0xFF;
    sm_out[idx*3+1] = (sm24 >> 8 ) & 0xFF;
    sm_out[idx*3+2] = (sm24 >> 16) & 0xFF;
}

template<int N>
__global__ void unpack_bf16_kernel(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   uint16_t* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint8_t  sm   = sm_in[idx];
    uint16_t sign = (sm >> 7) & 0x1;
    uint16_t mant =  sm & 0x7F;
    uint16_t bits = (sign << 15) |
                    (static_cast<uint16_t>(exp_in[idx]) << 7) |
                    mant;
    out[ix.offset(idx)] = bits;
}

template<int N>
__global__ void unpack_fp32_kernel(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   float* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    uint32_t sm24 =  sm_in[idx*3+0]
                   | (sm_in[idx*3+1] << 8 )
                   | (sm_in[idx*3+2] << 16);

    uint32_t sign = (sm24 >> 23) & 0x1;
    uint32_t mant =  sm24 & 0x7FFFFF;

    uint32_t bits = (sign << 31)
                  | (static_cast<uint32_t>(exp_in[idx]) << 23)
                  |  mant;
    reinterpret_cast<uint32_t*>(out)[ ix.offset(idx) ] = bits;
}

//----------------------------------------------------------------------------
//  host wrappers
//----------------------------------------------------------------------------
static inline dim3 grid_for(int64_t n) { return dim3((n + 255) / 256); }

std::vector<at::Tensor> pack_tensor(const at::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(t.scalar_type()==at::kFloat || t.scalar_type()==at::kBFloat16,
                "dtype must be fp32 / bf16");
    const int64_t N = t.numel();
    dim3 grid = grid_for(N);
    // auto exp = at::empty_like(t, t.options().dtype(at::kByte));       // [N]
    auto exp = at::empty(N, t.options().dtype(at::kByte));

    if (t.scalar_type() == at::kFloat) {
        auto sm  = at::empty({N,3}, t.options().dtype(at::kByte));    // [N,3]
        auto ix  = make_indexer<4>(t);
        pack_fp32_kernel<4><<<grid,256>>>(
            t.data_ptr<float>(), ix,
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(), N);
        return {exp, sm};
    } else {    // bf16
        // auto sm  = at::empty_like(t, at::dtype(at::kByte));           // [N]
        auto sm = at::empty(N, t.options().dtype(at::kByte));
        auto ix  = make_indexer<4>(t);
        pack_bf16_kernel<4><<<grid,256>>>(
            reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
            ix,
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(), N);
        return {exp, sm};
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
                         c10::ScalarType dtype) {

    TORCH_CHECK(exp.is_cuda() && sm.is_cuda(), "inputs must be CUDA uint8");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16, "dtype mismatch");
    
    const int64_t need = required_storage(storage_offset, sizes, strides);

    const int64_t N = exp.numel();
    dim3 grid = grid_for(N);

    auto flat = at::empty({need}, exp.options().dtype(dtype));   // ① 先建 1-D storage
    auto out  = flat.as_strided(sizes, strides, storage_offset); // ② 再做 view

    // // 1) allocate output with *original* layout
    // auto out = at::empty_strided(sizes, strides,
    //                              exp.options().dtype(dtype));
    // if (storage_offset != 0)
    //     out = out.as_strided(sizes, strides, storage_offset);

    auto ix = make_indexer<4>(out);

    // 2) launch kernel
    if (dtype == at::kFloat) {
        unpack_fp32_kernel<4><<<grid,256>>>(
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(),
            ix,
            out.data_ptr<float>(),
            N);
    } else {
        unpack_bf16_kernel<4><<<grid,256>>>(
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(),
            ix,
            reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
            N);
    }
    return out;
}

//----------------------------------------------------------------------------
//  pybind
//----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split", &pack_tensor, "pack tensor -> (exp, sm)");
    m.def("merge", &unpack_tensor,
          "unpack (exp,sm,sizes,strides,offset,dtype) -> tensor",
          py::arg("exp"), py::arg("sm"),
          py::arg("sizes"), py::arg("strides"), py::arg("storage_offset"),
          py::arg("dtype"));
}
