
from __future__ import print_function, absolute_import
from pyramid.utils import get_random_state
from pyramid.compat.python import long
from numpy.random import RandomState
from nose.tools import assert_raises


def test_random_state():
    assert isinstance(get_random_state(1), RandomState)
    assert isinstance(get_random_state(None), RandomState)
    assert isinstance(get_random_state(long(1)), RandomState)
    assert isinstance(get_random_state(RandomState(None)), RandomState)

    # show fails for string
    assert_raises(ValueError, get_random_state, 'some_seed')
