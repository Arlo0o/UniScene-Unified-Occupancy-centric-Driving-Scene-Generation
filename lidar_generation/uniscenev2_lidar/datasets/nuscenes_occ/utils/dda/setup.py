from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dda3d_cuda',
    ext_modules=[
        CUDAExtension(
            name="dda3d_gpu",  # 生成的模块名
            sources=['dda3d.cpp', 'dda3d_cuda.cu']
            # extra_compile_args={
            #     'cxx': [],
            #     'nvcc': ['-arch=sm_60']  # 根据你的 GPU 架构调整
            # },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)