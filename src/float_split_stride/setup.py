from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

setup(
    name="float_split_stride",
    ext_modules=[
        CUDAExtension(
            "float_split_stride._ext",
            ["float_split_stride.cu"],
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                # 多卡/多平台可改用 $TORCH_CUDA_ARCH_LIST
            ],
        )
    ],
    packages=["float_split_stride"],
    cmdclass={"build_ext": BuildExtension},
)
