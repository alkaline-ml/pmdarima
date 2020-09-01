# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# The pmdarima module

from pathlib import Path
import os as _os
import sys as _sys
import warnings as _warnings

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
# We only create a VERSION file in CI/CD on tagged commits.
# For local development, or non-tagged commits, we will use 0.0.0
try:
    version_path = Path(__file__).parent / 'VERSION'
    __version__ = version_path.read_text().strip()
except FileNotFoundError:
    __version__ = '0.0.0'

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
    from .arima import auto_arima, ARIMA, AutoARIMA, StepwiseContext, decompose
    from .utils import acf, autocorr_plot, c, pacf, plot_acf, plot_pacf, \
        tsdisplay
    from .utils._show_versions import show_versions

    # Need these namespaces at the top so they can be used like:
    # pm.datasets.load_wineind()
    from . import arima
    from . import compat
    from . import context_managers
    from . import datasets
    from . import decorators
    from . import model_selection
    from . import preprocessing
    from . import utils

    __all__ = [
        # Namespaces we want exposed at top:
        'arima',
        'compat',
        'context_managers',
        'datasets',
        'decorators',
        'model_selection',
        'preprocessing',
        'utils',

        # Functions & non-modules at top level
        'ARIMA',
        'acf',
        'autocorr_plot',
        'auto_arima',
        'c',
        'decompose',
        'pacf',
        'plot_acf',
        'plot_pacf',
        'show_versions',
        'StepwiseContext',
    ]

    _min_version = (3, 6)
    _py_version = _sys.version_info
    if _py_version < _min_version:
        _warnings.warn(
            "pmdarima is not built or tested against versions of python "
            "older than {0}.{1}. Your python version ({2}.{3}.{4}) is "
            "not guaranteed to be supported".format(
                _min_version[0], _min_version[1],
                _py_version.major, _py_version.minor, _py_version.micro,
            )
        )

    # Delete unwanted variables from global
    del _min_version
    del _os
    del _py_version
    del _sys
    del _warnings
    del __check_build
    del __PMDARIMA_SETUP__


def setup_module(module):
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
