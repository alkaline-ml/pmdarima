"""
Minimal setup.py for building pmdarima Cython extensions.
All package metadata is now in pyproject.toml.
"""
import os
import sys
from setuptools import setup, Extension
import numpy

# Get BLAS info for extensions that need it
def get_blas_info():
    """Get BLAS library information for linking Cython extensions."""
    try:
        from numpy.distutils.system_info import get_info
        blas_info = get_info('blas_opt', 0)

        def atlas_not_found(blas_info_):
            def_macros = blas_info_.get('define_macros', [])
            for x in def_macros:
                if x[0] == "NO_ATLAS_INFO":
                    return True
                if x[0] == "ATLAS_INFO" and "None" in str(x[1]):
                    return True
            return False

        if not blas_info or atlas_not_found(blas_info):
            cblas_libs = ['cblas']
            libraries = []
        else:
            cblas_libs = blas_info.pop('libraries', [])
            libraries = cblas_libs.copy()

        # Add math library on POSIX systems
        if os.name == 'posix':
            libraries.append('m')

        include_dirs = blas_info.pop('include_dirs', [])
        extra_compile_args = blas_info.pop('extra_compile_args', [])

        return {
            'libraries': libraries,
            'include_dirs': include_dirs,
            'extra_compile_args': extra_compile_args,
            **blas_info
        }
    except (ImportError, AttributeError):
        # If numpy.distutils is not available, use minimal configuration
        libraries = ['m'] if os.name == 'posix' else []
        return {
            'libraries': libraries,
            'include_dirs': [],
            'extra_compile_args': []
        }

# Get numpy include directory
include_dirs = [numpy.get_include()]

# Get BLAS configuration for extensions that need it
blas_info = get_blas_info()

# Combine include directories
blas_include_dirs = include_dirs + blas_info['include_dirs']

# Define extensions
ext_modules = [
    # Simple extension without BLAS
    Extension(
        name='pmdarima.__check_build._check_build',
        sources=['pmdarima/__check_build/_check_build.pyx'],
        include_dirs=include_dirs
    ),
    # Extensions that need BLAS linking
    Extension(
        name='pmdarima.arima._arima',
        sources=['pmdarima/arima/_arima.pyx'],
        include_dirs=blas_include_dirs,
        libraries=blas_info['libraries'],
        extra_compile_args=blas_info['extra_compile_args'],
    ),
    Extension(
        name='pmdarima.preprocessing.exog._fourier',
        sources=['pmdarima/preprocessing/exog/_fourier.pyx'],
        include_dirs=blas_include_dirs,
        libraries=blas_info['libraries'],
        extra_compile_args=blas_info['extra_compile_args'],
    ),
    Extension(
        name='pmdarima.utils._array',
        sources=['pmdarima/utils/_array.pyx'],
        include_dirs=blas_include_dirs,
        libraries=blas_info['libraries'],
        extra_compile_args=blas_info['extra_compile_args'],
    ),
]

# Setup with extensions only - metadata comes from pyproject.toml
setup(ext_modules=ext_modules)
