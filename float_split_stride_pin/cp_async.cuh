#pragma once
#include <cuda.h>

// comment this line if your GPU ≥ SM80
// #define NO_CP_ASYNC

#ifndef NO_CP_ASYNC
template<int BYTES>
__device__ __forceinline__ void cp_async(void* dst, const void* src) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16,
                  "cp.async supports 4/8/16-byte only");
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :: "r"(dst),                // 64-bit register
           "r"(src),                // 64-bit register
           "n"(BYTES));            // immed-constant
}
#else
template<int BYTES>
__device__ __forceinline__ void cp_async(void*, const void*) {}
#endif
