# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Random utils for pyramid

from __future__ import absolute_import, division, print_function
from numpy.random import RandomState

# long is not defined in python 3
from ..compat.python import long

__all__ = [
    'get_random_state'
]


def get_random_state(random_state):
    """Get a ``numpy.random.RandomState`` PRNG given a seed or
    an existing ``RandomState``. This method is lifted from one of
    my other repos, https://github.com/tgsmith61591/smrt, to avoid
    multiple dependencies...

    Parameters
    ----------
    random_state : ``RandomState``, int or None
        The seed or PRNG.
    """
    if random_state is None:
        return RandomState()
    elif isinstance(random_state, RandomState):
        return random_state
    elif isinstance(random_state, (int, long)):
        return RandomState(random_state)
    else:
        raise ValueError('Cannot seed RandomState given class=%s' % type(random_state))
