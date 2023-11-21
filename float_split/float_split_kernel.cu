#include <torch/extension.h>
#include <cuda_fp16.h>


inline dim3 make_grids(size_t N, int threads = 256) {
    return dim3((N + threads - 1) / threads);
}

//---------------------------------------------------------------------
//  ① FP32  →  exp_u8  +  sm_u8[3]
//---------------------------------------------------------------------
__global__
void split_fp32_kernel(const float* __restrict__ in,
                      uint8_t* __restrict__ exp_out,
                      uint8_t* __restrict__ sm_out,
                      size_t N) {

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint32_t bits = reinterpret_cast<const uint32_t*>(in)[idx];
    uint8_t exp   = (bits >> 23) & 0xFF;
    uint32_t sm24 = ((bits & 0x7FFFFF)      ) |  // mantissa
                    ((bits >> 8) & 0x800000);    // sign<<23

    exp_out[idx]            = exp;
    sm_out[idx * 3 + 0]     =  sm24        & 0xFF;
    sm_out[idx * 3 + 1]     = (sm24 >>  8) & 0xFF;
    sm_out[idx * 3 + 2]     = (sm24 >> 16) & 0xFF;
}



//---------------------------------------------------------------------
//  ② BF16  →  exp_u8  +  sm_u8
//---------------------------------------------------------------------
__global__
void split_bf16_kernel(const uint16_t* __restrict__ in,   // 直接按 uint16_t 读取
                      uint8_t*  __restrict__ exp_out,
                      uint8_t*  __restrict__ sm_out,
                      size_t N) {
    
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint16_t bits = in[idx];
    exp_out[idx]  = (bits >> 7) & 0xFF;
    sm_out[idx]   = ((bits >> 15) & 0x1) << 7      // sign
                  | (bits & 0x7F);                 // mantissa
}


//---------------------------------------------------------------------
//  ③ exp_u8 + sm_u8[3]  →  FP32
//---------------------------------------------------------------------
__global__
void merge_fp32_kernel(const uint8_t* __restrict__  exp_in,
                        const uint8_t* __restrict__  sm_in,
                        float*      __restrict__  out,
                        size_t N) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint32_t sm24 =  sm_in[idx * 3 + 0]
                   | (sm_in[idx * 3 + 1] << 8)
                   | (sm_in[idx * 3 + 2] << 16);

    uint32_t sign = (sm24 >> 23) & 0x1;
    uint32_t mant =  sm24 & 0x7FFFFF;
    uint32_t bits = (sign << 31)
                  | (static_cast<uint32_t>(exp_in[idx]) << 23)
                  |  mant;

    reinterpret_cast<uint32_t*>(out)[idx] = bits;
}

//---------------------------------------------------------------------
//  ④ exp_u8 + sm_u8  →  BF16
//---------------------------------------------------------------------
__global__
void merge_bf16_kernel(const uint8_t* __restrict__ exp_in,
                        const uint8_t* __restrict__ sm_in,
                        uint16_t*    __restrict__ out,
                        size_t N) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint8_t  sm   = sm_in[idx];
    uint16_t sign = (sm >> 7) & 0x1;
    uint16_t mant =  sm & 0x7F;

    uint16_t bits = (sign << 15)
                  | (static_cast<uint16_t>(exp_in[idx]) << 7)
                  |  mant;

    out[idx] = bits;
}


//---------------------------------------------------------------------
//  PyTorch C++ API 封装
//---------------------------------------------------------------------
std::vector<at::Tensor> split_tensor(at::Tensor t) {
    TORCH_CHECK(t.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), "input must be contiguous");

    const auto N = t.numel();
    const dim3 blocks = make_grids(N);

    if (t.scalar_type() == at::kFloat) {
        auto exp  = at::empty_like(t, at::dtype(at::kByte));           // [N]
        auto sm   = at::empty({N, 3}, t.options().dtype(at::kByte));   // [N,3]

        split_fp32_kernel<<<blocks, 256>>>(
            t.data_ptr<float>(),
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(),
            N);
        return {exp, sm};
    }
    if (t.scalar_type() == at::kBFloat16) {
        auto exp = at::empty_like(t, at::dtype(at::kByte));            // [N]
        auto sm  = at::empty_like(t, at::dtype(at::kByte));            // [N]

        split_bf16_kernel<<<blocks, 256>>>(
            reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(),
            N);
        return {exp, sm};
    }
    TORCH_CHECK(false, "Unsupported dtype");
}

at::Tensor merge_tensor(at::Tensor exp,
                         at::Tensor sm,
                         c10::ScalarType dtype) {

    TORCH_CHECK(exp.is_cuda() && sm.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(exp.is_contiguous() && sm.is_contiguous(), "inputs must be contiguous");

    const auto N = exp.numel();
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16, "dtype must be float32 or bfloat16");

    const dim3 blocks = make_grids(N);

    if (dtype == at::kFloat) {
        auto out = at::empty_like(exp, at::dtype(at::kFloat));         // [N] float32
        merge_fp32_kernel<<<blocks, 256>>>(
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(),
            out.data_ptr<float>(),
            N);
        return out.view_as(exp);
    } else { // BF16
        auto out = at::empty_like(exp, at::dtype(at::kBFloat16));      // [N] bf16
        merge_bf16_kernel<<<blocks, 256>>>(
            exp.data_ptr<uint8_t>(),
            sm.data_ptr<uint8_t>(),
            reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
            N);
        return out.view_as(exp);
    }
}

//---------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split",   &split_tensor,   "Pack float/bf16 -> (exp, sm)");
    m.def("merge", &merge_tensor, "Unpack (exp, sm, dtype) -> tensor");
}