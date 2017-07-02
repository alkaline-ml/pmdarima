# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The pyramid module

__version__ = '0.5-dev3'

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
    __all__ = [
        'arima',
        'compat',
        'utils'
    ]

    # top-level imports - if any
    # todo


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
