# -*- coding: utf-8 -*-
#
# Meta testing tests

from __future__ import absolute_import

from pyramid.utils.testing import assert_raises


def test_assert_raises():

    def raise_value_error():
        raise ValueError()

    def raise_type_error():
        raise TypeError()

    def innocuous_function():
        pass

    # Show it works when it receives the expected error
    assert_raises(ValueError, raise_value_error)  # Should pass
    assert_raises(TypeError, raise_type_error)    # Should pass

    # Show we fail when we get the wrong error types
    assert_raises(TypeError,
                  lambda: assert_raises(ValueError, raise_type_error))

    assert_raises(ValueError,
                  lambda: assert_raises(TypeError, raise_value_error))

    # And show we get an assertion error when it doesn't raise at all
    assert_raises(AssertionError,
                  lambda: assert_raises(ValueError, innocuous_function))
