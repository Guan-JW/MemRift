from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

SRC = ["float_split_kernel.cu"]
ARCH = "compute_87,code=sm_87"       # Jetson Orin；PC 可写 80/86 等

setup(
    name="float_split",
    version="0.1.0",
    packages=["float_split"],
    ext_modules=[
        CUDAExtension(
            "float_split._ext",
            SRC,
            extra_cuda_cflags=[
                "-O3", "-lineinfo",
                f"-gencode=arch={ARCH}"
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
