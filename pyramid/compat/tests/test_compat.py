
from __future__ import absolute_import
from pyramid.compat.python import long, xrange


def test_compatibility():
    for _ in xrange(5):
        pass

    assert long(5) > 4  # show this works on both
