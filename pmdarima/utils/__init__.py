# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from .array import *
from .metaestimators import *
from .testing import *
from .visualization import *
from .wrapped import *


def get_callable(key, dct):
    """Get the callable mapped by a key from a dictionary. This is
    necessary for pickling (so we don't try to pickle an unbound method).

    Parameters
    ----------
    key : str
        The key for the ``dct`` dictionary.

    dct : dict
        The dictionary of callables.
    """
    from sklearn.externals import six
    fun = dct.get(key, None)

    if not isinstance(key, six.string_types) or \
            fun is None:  # ah, that's no fun :(
        raise ValueError('key must be a string in one in %r, but got %r'
                         % (dct, key))
    return fun


__all__ = [s for s in dir() if not s.startswith("_")]
