# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Patch backend for MPL

import sys
import os

__all__ = [
    'get_compatible_pyplot',
    'mpl_hist_arg'
]


def get_compatible_pyplot(backend=None, debug=True):
    """Make the backend of MPL compatible.

    In Travis Mac distributions, python is not installed as a framework. This
    means that using the TkAgg backend is the best solution (so it doesn't
    try to use the mac OS backend by default).

    Parameters
    ----------
    backend : str, optional (default="TkAgg")
        The backend to default to.

    debug : bool, optional (default=True)
        Whether to log the existing backend to stderr.
    """
    import matplotlib

    # If the backend provided is None, just default to
    # what's already being used.
    existing_backend = matplotlib.get_backend()
    if backend is not None:
        # Can this raise?...
        matplotlib.use(backend)

        # Print out the new backend
        if debug:
            sys.stderr.write("Currently using '%s' MPL backend, "
                             "switching to '%s' backend%s"
                             % (existing_backend, backend, os.linesep))

    # If backend is not set via env variable, but debug is
    elif debug:
        sys.stderr.write("Using '%s' MPL backend%s"
                         % (existing_backend, os.linesep))

    from matplotlib import pyplot as plt
    return plt


def mpl_hist_arg(value=True):
    """Find the appropriate `density` kwarg for our given matplotlib version.

    This will determine if we should use `normed` or `density`. Additionally,
    since this is a kwarg, the user can supply a value (True or False) that
    they would like in the output dictionary.

    Parameters
    ----------
    value : bool, optional (default=True)
        The boolean value of density/normed

    Returns
    -------
    density_kwarg : dict
        A dictionary containing the appropriate density kwarg for the
        installed  matplotlib version, mapped to the provided or default
        value
    """
    import matplotlib

    density_kwarg = 'density' if matplotlib.__version__ >= '2.1.0'\
        else 'normed'
    return {density_kwarg: value}
