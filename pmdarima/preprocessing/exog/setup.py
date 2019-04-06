# -*- coding: utf-8 -*-

import os

import numpy
from numpy.distutils.misc_util import Configuration

from pmdarima._build_utils import get_blas_info


def configuration(parent_package="", top_path=None):
    cblas_libs, blas_info = get_blas_info()

    # Use this rather than cblas_libs so we don't fail on Windows
    libraries = []
    if os.name == 'posix':
        cblas_libs.append('m')
        libraries.append('m')

    config = Configuration("exog", parent_package, top_path)
    config.add_extension("_fourier",
                         sources=["_fourier.pyx"],
                         include_dirs=[numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         libraries=libraries,
                         extra_compile_args=blas_info.pop(
                             'extra_compile_args', []),
                         **blas_info)

    config.add_subpackage('tests')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
