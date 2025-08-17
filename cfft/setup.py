from setuptools import setup, Extension
import sys

extra_compile_args = []
extra_link_args = []

if sys.platform.startswith('win'):
    # Optimize and enable fast math; MSVC flags
    extra_compile_args = ['/O2']
else:
    # GCC/Clang flags
    extra_compile_args = ['-O3', '-ffast-math']

module = Extension(
    'cfft',
    sources=['cfft.c'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='cfft',
    version='0.1.0',
    description='Simple C FFT module (radix-2 Cooley-Tukey) for real-valued power spectrum',
    ext_modules=[module],
)


