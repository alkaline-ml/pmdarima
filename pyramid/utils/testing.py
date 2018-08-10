# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from __future__ import absolute_import

__all__ = [
    'assert_raises'
]


def assert_raises(exception_type, func, *args, **kwargs):
    """Assert that a function raises an exception.

    This function is a testing utility that asserts that a function
    will raise a given exception. If it does not, it will raise an
    AssertionError. Alternatively, if it raises a separate exception,
    it will raise *that* exception.

    Parameters
    ----------
    exception_type : BaseException or BaseError
        The exception type

    func : callable
        The function that is expected to raise

    Notes
    -----
    This is roughly equivalent to the ``nose.tools.assert_raises`` utility,
    but since the nose package has been deprecated and we favor pytest,
    we provide this here to avoid another dependency.

    This was adopted from another of my packages,
    `skoot <https://github.com/tgsmith61591/skoot>`_.

    Examples
    --------
    >>> def function_that_raises():
    ...     raise ValueError("boo!")
    >>> assert_raises(ValueError, function_that_raises)
    """
    try:
        func(*args, **kwargs)
    # except only the prescribed type
    except exception_type:
        pass
    # anything else raises
    except Exception:
        raise
    # otherwise we got nothing
    else:
        raise AssertionError("%s did not raise %r"
                             % (func.__name__, exception_type))
