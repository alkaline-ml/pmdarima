# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup for submodules of pyramid

from __future__ import absolute_import
import os


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('pyramid', parent_package, top_path)

    # modules
    config.add_subpackage('stats')
    config.add_subpackage('stats/tests')
    config.add_subpackage('utils')
    config.add_subpackage('utils/tests')

    # modules with cython
    config.add_subpackage('arima')
    config.add_subpackage('arima/tests')

    # misc repo tests
    config.add_subpackage('tests')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
