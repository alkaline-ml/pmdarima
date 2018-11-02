# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The pyramid module

import os as _os

__version__ = "0.9.0"

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

    # Warn for Issue #34 (https://github.com/tgsmith61591/pyramid/issues/34)
    import warnings
    warnings.warn("""
    The 'pyramid' package will be migrating to a new namespace beginning in 
    version 1.0.0: 'pmdarima'. This is due to a package name collision with the
    Pyramid web framework. For more information, see Issue #34:
    
        https://github.com/tgsmith61591/pyramid/issues/34
        
    The package will subsequently be installable via the name 'pmdarima'; the
    only functional change to the user will be the import name. All imports
    from 'pyramid' will change to 'pmdarima'.
    """, UserWarning)

    # On first import, check the cache, warn if needed
    from ._config import _warn_for_cache_size
    _warn_for_cache_size()

    # Delete unwanted variables from global
    del _os
    del warnings
    # del _config  # don't delete in case user wants to amend it at top level
    del _warn_for_cache_size
    del __check_build
    del __PYRAMID_SETUP__


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
