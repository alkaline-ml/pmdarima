# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup for submodules of pyramid

from __future__ import absolute_import

import os

from pyramid._build_utils import maybe_cythonize_extensions


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('pyramid', parent_package, top_path)

    # build utilities
    config.add_subpackage('__check_build')
    config.add_subpackage('__check_build/tests')
    config.add_subpackage('_build_utils')
    config.add_subpackage('_build_utils/tests')

    # modules
    config.add_subpackage('compat')
    config.add_subpackage('compat/tests')
    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    # the following packages have cython, so they have to be build
    # after the above.
    config.add_subpackage('arima')
    config.add_subpackage('arima/tests')

    # do cythonization
    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
