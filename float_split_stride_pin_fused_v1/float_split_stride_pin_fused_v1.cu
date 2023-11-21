#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>          // at::BFloat16 helpers
#include <cstdint>

//----------------------------------------------------------------------------
//  utilities: indexer <sizes,strides,offset>  →  flatten / offset
//----------------------------------------------------------------------------
#define GRID_STRIDE_LOOP(idx, N) \
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < (N); idx += blockDim.x * gridDim.x)

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
template<int N, int VEC = 8>
__global__ void pack_bf16_kernel_vec(
        const uint16_t* __restrict__ in,
        StridedIndex<N> ix,
        uint8_t* __restrict__ exp_out,
        uint8_t* __restrict__ sm_out,
        int64_t numel)
{
    int64_t num8 = numel / VEC;
    GRID_STRIDE_LOOP(i8, num8) {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int64_t idx = i8 * VEC + j;
            uint16_t bits = in[ix.offset(idx)];
            exp_out[idx] = (bits >> 7) & 0xFF;
            sm_out[idx] = ((bits >> 15) & 0x1) << 7 | (bits & 0x7F);
        }
    }
    // tail
    int tail = numel % VEC;
    int64_t base = numel - tail;
    if (threadIdx.x == 0 && tail) {
        for (int k = 0; k < tail; ++k) {
            int64_t idx = base + k;
            uint16_t bits = in[ix.offset(idx)];
            exp_out[idx] = (bits >> 7) & 0xFF;
            sm_out[idx] = ((bits >> 15) & 0x1) << 7 | (bits & 0x7F);
        }
    }
}


template<int N, int VEC = 4>
__global__ void pack_fp32_kernel_vec(
        const float* __restrict__ in,
        StridedIndex<N> ix,
        uint8_t* __restrict__ exp_out,
        uint8_t* __restrict__ sm_out,
        int64_t numel)
{
    int64_t num4 = numel / VEC;
    GRID_STRIDE_LOOP(i4, num4) {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int64_t idx = i4 * VEC + j;
            uint32_t bits = reinterpret_cast<const uint32_t*>(in)[ix.offset(idx)];
            exp_out[idx] = (bits >> 23) & 0xFF;
            uint32_t sm24 = (bits & 0x7FFFFF) | ((bits >> 8) & 0x800000);
            sm_out[idx * 3 + 0] = sm24 & 0xFF;
            sm_out[idx * 3 + 1] = (sm24 >> 8) & 0xFF;
            sm_out[idx * 3 + 2] = (sm24 >> 16) & 0xFF;
        }
    }
    // handle tail
    int tail = numel % VEC;
    int64_t base = numel - tail;
    if (threadIdx.x == 0 && tail) {
        for (int k = 0; k < tail; ++k) {
            int64_t idx = base + k;
            uint32_t bits = reinterpret_cast<const uint32_t*>(in)[ix.offset(idx)];
            exp_out[idx] = (bits >> 23) & 0xFF;
            uint32_t sm24 = (bits & 0x7FFFFF) | ((bits >> 8) & 0x800000);
            sm_out[idx * 3 + 0] = sm24 & 0xFF;
            sm_out[idx * 3 + 1] = (sm24 >> 8) & 0xFF;
            sm_out[idx * 3 + 2] = (sm24 >> 16) & 0xFF;
        }
    }
}


template<int N, int VEC = 8>
__global__ void unpack_bf16_kernel_vec(
        const uint8_t* __restrict__ exp_in,
        const uint8_t* __restrict__ sm_in,
        StridedIndex<N> ix,
        uint16_t* __restrict__ out,
        int64_t numel)
{
    int64_t num8 = numel / VEC;
    GRID_STRIDE_LOOP(i8, num8) {
        // 向量化载入 8 个 exp/sm
        uint64_t exps = *(reinterpret_cast<const uint64_t*>(exp_in + i8 * VEC));
        uint64_t sms  = *(reinterpret_cast<const uint64_t*>(sm_in + i8 * VEC));

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int64_t idx = i8 * VEC + j;
            uint8_t exp = (exps >> (8 * j)) & 0xFF;
            uint8_t sm  = (sms  >> (8 * j)) & 0xFF;
            
            uint16_t sign = (sm >> 7) & 0x1;
            uint16_t mant = sm & 0x7F;
            uint16_t bits = (sign << 15) | (static_cast<uint16_t>(exp) << 7) | mant;
            out[ix.offset(idx)] = bits;
        }
    }
    // tail
    int tail = numel % VEC;
    int64_t base = numel - tail;
    if (threadIdx.x == 0 && tail) {
        for (int k = 0; k < tail; ++k) {
            int64_t idx = base + k;
            uint8_t sm = sm_in[idx];
            uint16_t sign = (sm >> 7) & 0x1;
            uint16_t mant = sm & 0x7F;
            uint16_t bits = (sign << 15) | (static_cast<uint16_t>(exp_in[idx]) << 7) | mant;
            out[ix.offset(idx)] = bits;
        }
    }
}


template<int N, int VEC = 4>
__global__ void unpack_fp32_kernel_vec(
        const uint8_t* __restrict__ exp_in,
        const uint8_t* __restrict__ sm_in,
        StridedIndex<N> ix,
        float* __restrict__ out,
        int64_t numel)
{
    int64_t num4 = numel / VEC;
    GRID_STRIDE_LOOP(i4, num4) {
        // 向量化载入 4 个 exp
        uint32_t exps = *(reinterpret_cast<const uint32_t*>(exp_in + i4 * VEC));
        // 向量化载入 12 个 sm
        const uint8_t* sm_ptr = sm_in + i4 * VEC * 3;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int64_t idx = i4 * VEC + j;
            uint8_t exp = (exps >> (8 * j)) & 0xFF;
            uint32_t sm24 = sm_ptr[j * 3 + 0]
                          | (sm_ptr[j * 3 + 1] << 8)
                          | (sm_ptr[j * 3 + 2] << 16);
            uint32_t bits = ((sm24 >> 23) & 0x1) << 31
                                        | (static_cast<uint32_t>(exp) << 23)
                                        | (sm24 & 0x7FFFFF);
            reinterpret_cast<uint32_t*>(out)[ix.offset(idx)] = bits;
        }
    }
    // tail
    int tail = numel % VEC;
    int64_t base = numel - tail;
    if (threadIdx.x == 0 && tail) {
        for (int k = 0; k < tail; ++k) {
            int64_t idx = base + k;
            uint32_t sm24 = sm_in[idx * 3 + 0]
                          | (sm_in[idx * 3 + 1] << 8)
                          | (sm_in[idx * 3 + 2] << 16);
            uint32_t bits = ((sm24 >> 23) & 0x1) << 31
                          | (static_cast<uint32_t>(exp_in[idx]) << 23)
                          | (sm24 & 0x7FFFFF);
            reinterpret_cast<uint32_t*>(out)[ix.offset(idx)] = bits;
        }
    }
}


//----------------------------------------------------------------------------
//  host wrappers
//----------------------------------------------------------------------------
static inline dim3 grid_for(int64_t n, int block = 256, int VEC = 4) {
     int n_vec = (n + VEC - 1) / VEC;
     return dim3((n_vec + block - 1) / block); 
}

std::vector<at::Tensor> pack_tensor(const at::Tensor& t, unsigned long long stream_ptr) {
    TORCH_CHECK(t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, "dtype must be fp32 / bf16");

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    const int64_t N = t.numel();
    int block = 256;
    // auto exp = at::empty(N, at::dtype(at::kByte).device(t.device()));
    auto exp = at::empty(N, at::dtype(at::kByte).pinned_memory(true));
    auto ix = make_indexer<4>(t);

    at::Tensor sm;

    if (t.scalar_type() == at::kFloat) {
        sm = at::empty({N, 3}, at::dtype(at::kByte).device(t.device()));
        const int VEC = 4;
        dim3 grid = grid_for(N, block, VEC);
        pack_fp32_kernel_vec<4, 4><<<grid, block, 0, stream>>>(
            t.data_ptr<float>(), ix, exp.data_ptr<uint8_t>(), sm.data_ptr<uint8_t>(), N);
    } else {
        sm = at::empty(N, at::dtype(at::kByte).device(t.device()));
        const int VEC = 8;
        dim3 grid = grid_for(N, block, VEC);
        pack_bf16_kernel_vec<4, VEC><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()), ix,
            exp.data_ptr<uint8_t>(), sm.data_ptr<uint8_t>(), N);
    }
    return {exp, sm};
}

int64_t required_storage(int64_t base_off,
            const std::vector<int64_t>& sizes,
            const std::vector<int64_t>& strides) {
    int64_t max_off = base_off;
    for (size_t i = 0; i < sizes.size(); ++i)
        max_off += (sizes[i] - 1) * strides[i];
    return max_off + 1;
}

at::Tensor unpack_tensor(at::Tensor exp,
                         at::Tensor sm,
                         std::vector<int64_t> sizes,
                         std::vector<int64_t> strides,
                         int64_t storage_offset,
                         c10::ScalarType dtype) {
    TORCH_CHECK(sm.is_cuda(), "sm must be CUDA uint8");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16, "dtype mismatch");

    int64_t N = exp.numel();
    int block = 256;
    const int64_t need = required_storage(storage_offset, sizes, strides);
    
    auto flat = at::empty({need}, sm.options().dtype(dtype));   // ① 先建 1-D storage
    auto out  = flat.as_strided(sizes, strides, storage_offset); // ② 再做 view
    auto ix = make_indexer<4>(out);

    if (dtype == at::kFloat) {
        const int VEC = 4;
        dim3 grid = grid_for(N, block, VEC);
        unpack_fp32_kernel_vec<4, VEC><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            exp.data_ptr<uint8_t>(), sm.data_ptr<uint8_t>(), ix, out.data_ptr<float>(), N);
    } else {
        const int VEC = 8;
        dim3 grid = grid_for(N, block, VEC);
        unpack_bf16_kernel_vec<4, VEC><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            exp.data_ptr<uint8_t>(), sm.data_ptr<uint8_t>(), ix,
            reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()), N);
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