# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The pmdarima module

import os as _os

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "1.3.1.dev0"

try:
    # this var is injected in the setup build to enable
    # the retrieval of the version number without actually
    # importing the un-built submodules.
    __PMDARIMA_SETUP__
except NameError:
    __PMDARIMA_SETUP__ = False

if __PMDARIMA_SETUP__:
    import sys
    sys.stderr.write('Partial import of pmdarima during the build process.%s'
                     % _os.linesep)
else:
    # check that the build completed properly. This prints an informative
    # message in the case that any of the C code was not properly compiled.
    from . import __check_build

    # Stuff we want at top-level
    from .arima import auto_arima, ARIMA, AutoARIMA
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

    # Delete unwanted variables from global
    del _os
    del __check_build
    del __PMDARIMA_SETUP__


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
