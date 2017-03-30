
from __future__ import print_function, absolute_import
from pyramid.utils import get_random_state
from numpy.random import RandomState
import sys

if sys.version_info[0] >= 3:
    long = int


def test_random_state():
    assert isinstance(get_random_state(1), RandomState)
    assert isinstance(get_random_state(None), RandomState)
    assert isinstance(get_random_state(long(1)), RandomState)
    assert isinstance(get_random_state(RandomState(None)), RandomState)
