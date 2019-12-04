# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Setup for submodules of pmdarima

import os
import sys

from pmdarima._build_utils import cythonize_extensions


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('pmdarima', parent_package, top_path)

    # build utilities
    config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils')

    # submodules that do NOT have their own setup.py. manually add their tests
    config.add_subpackage('compat')
    config.add_subpackage('compat/tests')
    config.add_subpackage('datasets')
    config.add_subpackage('datasets/tests')
    config.add_subpackage('model_selection')
    config.add_subpackage('model_selection/tests')

    # the following packages have cython or their own setup.py files.
    config.add_subpackage('arima')
    config.add_subpackage('preprocessing')
    config.add_subpackage('utils')

    # add test directory
    config.add_subpackage('tests')

    # Do cythonization, but only if this is not a release tarball, since the
    # C/C++ files are not necessarily forward compatible with future versions
    # of python.
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
