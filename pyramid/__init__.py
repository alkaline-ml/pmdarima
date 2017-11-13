# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The pyramid module

__version__ = "0.6.2"

try:
    # this var is injected in the setup build to enable
    # the retrieval of the version number without actually
    # importing the un-built submodules.
    __PYRAMID_SETUP__
except NameError:
    __PYRAMID_SETUP__ = False

if __PYRAMID_SETUP__:
    import sys
    import os
    sys.stderr.write('Partial import of pyramid during the build process.' + os.linesep)
else:
    # check that the build completed properly. This prints an informative
    # message in the case that any of the C code was not properly compiled.
    from . import __check_build

    __all__ = [
        'arima',
        'compat',
        'datasets',
        'utils'
    ]


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
