# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The pyramid module

import os as _os

__version__ = "0.9.0-dev"

try:
    # this var is injected in the setup build to enable
    # the retrieval of the version number without actually
    # importing the un-built submodules.
    __PYRAMID_SETUP__
except NameError:
    __PYRAMID_SETUP__ = False

if __PYRAMID_SETUP__:
    import sys
    sys.stderr.write('Partial import of pyramid during the build process.' +
                     _os.linesep)
else:
    # check that the build completed properly. This prints an informative
    # message in the case that any of the C code was not properly compiled.
    from . import __check_build

    # Stuff we want at top-level
    from .arima import auto_arima, ARIMA
    from .utils import acf, autocorr_plot, c, pacf, plot_acf, plot_pacf

    # Need these namespaces at the top so they can be used like:
    # pm.datasets.load_wineind()
    from . import arima
    from . import datasets
    from . import utils

    __all__ = [
        # Namespaces we want exposed at top:
        'arima',
        'compat',
        'datasets',
        'utils',

        # Function(s) at top level
        'ARIMA',
        'acf',
        'autocorr_plot',
        'auto_arima',
        'c',
        'pacf',
        'plot_acf',
        'plot_pacf'
    ]

    # On first import, check the cache, warn if needed
    from ._config import CACHE_WARN_BYTES, PYRAMID_ARIMA_CACHE as PAC
    from os.path import join as j, getsize as gf, isfile as isf
    import warnings

    cache_size = sum(gf(j(PAC, f)) for f in _os.listdir(PAC) if isf(j(PAC, f)))
    if cache_size > CACHE_WARN_BYTES:
        warnings.warn("The Pyramid cache ({0}) has grown to {1:,} bytes. "
                      "Consider cleaning out old ARIMA models or increasing "
                      "the max cache size in pyramid/_config.py (currently "
                      "{2:,} bytes) to avoid this warning in the future."
                      .format(PAC, cache_size, int(CACHE_WARN_BYTES)),
                      UserWarning)

    # Delete unwanted variables from global
    del _os
    del __check_build
    del __PYRAMID_SETUP__

    # Delete the variables we just created for the cache check
    del CACHE_WARN_BYTES
    del PAC
    del j
    del gf
    del isf
    del warnings
    del cache_size


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
