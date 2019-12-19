# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Patch backend for MPL

import sys
import os

__all__ = [
    'get_compatible_pyplot'
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
