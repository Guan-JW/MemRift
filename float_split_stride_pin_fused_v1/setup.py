from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="float_split_stride_pin_fused_v1",
    ext_modules=[
        CUDAExtension(
            "float_split_stride_pin_fused_v1._ext",
            ["float_split_stride_pin_fused_v1.cu"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                # 多卡/多平台可改用 $TORCH_CUDA_ARCH_LIST
            ],
        )
    ],
    packages=["float_split_stride_pin_fused_v1"],
    cmdclass={"build_ext": BuildExtension},
)
