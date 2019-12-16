# -*- coding: utf-8 -*-

import os

from numpy.distutils.misc_util import Configuration

from pmdarima._build_utils import get_blas_info


def configuration(parent_package="", top_path=None):
    cblas_libs, blas_info = get_blas_info()

    # Use this rather than cblas_libs so we don't fail on Windows
    libraries = []
    if os.name == 'posix':
        cblas_libs.append('m')
        libraries.append('m')

    config = Configuration("preprocessing", parent_package, top_path)

    config.add_subpackage('endog')
    config.add_subpackage('endog/tests')
    config.add_subpackage('exog')  # builds src and adds its own tests

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration().todict())
