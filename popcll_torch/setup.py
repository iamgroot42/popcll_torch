from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='popcll_torch',
    version='1.0',
    author='Anshuman Suri',
    author_email='anshuman@virginia.edu',
    ext_modules=[
        CUDAExtension('popcll_torch', [
            'popcll_torch.cpp',
            'popcll_torch_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })