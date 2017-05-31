import os
import os.path

import numpy
from numpy.distutils.misc_util import Configuration
from pyramid._build_utils import get_blas_info


def configuration(parent_package="", top_path=None):
    config = Configuration("arima", parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()
    if os.name == 'posix':
        cblas_libs.append('m')

    config.add_extension("_arima",
                         sources=["_arima.c"],
                         include_dirs=[numpy.get_include(),
                                       blas_info.pop('include_dirs', [])],
                         libraries=cblas_libs,
                         extra_compile_args=blas_info.pop('extra_compile_args', []),
                         **blas_info)
    config.add_subpackage('tests')

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
