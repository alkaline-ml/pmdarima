# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Patch backend for MPL

from __future__ import absolute_import

__all__ = [
    'get_compatible_pyplot'
]


def get_compatible_pyplot(default_backend="TkAgg"):
    """Make the backend of MPL compatible.

    In Travis Mac distributions, python is not installed as a framework. This
    means that using the TkAgg backend is the best solution (so it doesn't
    try to use the mac OS backend by default).

    Parameters
    ----------
    default_backend : str, optional (default="TkAgg")
        The backend to default to.
    """
    import matplotlib

    # Can this raise?...
    backend = matplotlib.get_backend()
    if backend == 'MacOSX':
        matplotlib.use(default_backend)

    from matplotlib import pyplot as plt
    return plt
