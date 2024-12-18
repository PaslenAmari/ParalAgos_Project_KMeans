# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import platform

# OpenMP on the platforms
if platform.system() == "Darwin":  # macOS
    os.environ["CC"] = "gcc-12"  # or appropriate version of gcc
    os.environ["CXX"] = "g++-12"
    openmp_args = ['-Xpreprocessor', '-fopenmp']
    openmp_libs = ['-lomp']
else:  # Linux and Windows
    openmp_args = ['-fopenmp']
    openmp_libs = ['-fopenmp']

# Define the extension
extensions = [
    Extension(
        "parallel_kmeans",
        ["src/parallel_kmeans.pyx"],
        extra_compile_args=openmp_args,
        extra_link_args=openmp_libs,
        include_dirs=[np.get_include()],
    )
]

# Setup configuration
setup(
    name="parallel_kmeans",
    version="1.0",
    description="Parallel K-means clustering implementation using OpenMP",
    packages=['src'],
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
    }),
    install_requires=[
        'numpy>=1.20.0',
        'cython>=0.29.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
    ],
    python_requires='>=3.7',
)
