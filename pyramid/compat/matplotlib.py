# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Patch backend for MPL

from __future__ import absolute_import

import sys
import os

__all__ = [
    'get_compatible_pyplot'
]


def get_compatible_pyplot(default_backend="TkAgg", debug=True,
                          replace=('MacOSX',)):
    """Make the backend of MPL compatible.

    In Travis Mac distributions, python is not installed as a framework. This
    means that using the TkAgg backend is the best solution (so it doesn't
    try to use the mac OS backend by default).

    Parameters
    ----------
    default_backend : str, optional (default="TkAgg")
        The backend to default to.

    debug : bool, optional (default=True)
        Whether to log the existing backend to stderr.

    replace : iterable, optional (default=('MacOSX',))
        The backends to replace.
    """
    import matplotlib

    # Can this raise?...
    backend = matplotlib.get_backend()
    if backend in replace:
        matplotlib.use(default_backend)

    # Print out the new backend
    if debug:
        sys.stderr.write("Using '%s' MPL backend%s"
                         % (matplotlib.get_backend(), os.linesep))
    from matplotlib import pyplot as plt
    return plt
