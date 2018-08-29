# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Metaestimators for the ARIMA class. These classes are derived from the
# sklearn metaestimators, but adapted for more specific use with pyramid.

from __future__ import absolute_import

from operator import attrgetter
from functools import update_wrapper

__all__ = [
    'if_has_delegate'
]


class _IffHasDelegate(object):
    """Implements a conditional property using the descriptor protocol.
    Using this class to create a decorator will raise an ``AttributeError``
    if none of the delegates (specified in ``delegate_names``) is an attribute
    of the base object or the first found delegate does not have an attribute
    ``attribute_name``.

    This allows ducktyping of the decorated method based on
    ``delegate.attribute_name``. Here ``delegate`` is the first item in
    ``delegate_names`` for which ``hasattr(object, delegate) is True``.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """
    def __init__(self, fn, delegate_names):
        self.fn = fn
        self.delegate_names = delegate_names

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def __get__(self, obj, type=None):
        # raise an AttributeError if the attribute is not present on the object
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            for delegate_name in self.delegate_names:
                try:
                    attrgetter(delegate_name)(obj)
                except AttributeError:
                    continue
                else:
                    break
            else:
                attrgetter(self.delegate_names[-1])(obj)

        # lambda, but not partial, allows help() to work with update_wrapper
        out = (lambda *args, **kwargs: self.fn(obj, *args, **kwargs))
        # update the docstring of the returned function
        update_wrapper(out, self.fn)
        return out


def if_has_delegate(delegate):
    """Wrap a delegated instance attribute function.

    Creates a decorator for methods that are delegated in the presence of a
    results wrapper. This enables duck-typing by ``hasattr`` returning True
    according to the sub-estimator.

    This function was adapted from scikit-learn, which defines
    ``if_delegate_has_method``, but operates differently by injecting methods
    not based on method presence, but by delegate presence.

    Examples
    --------
    >>> from pyramid.utils.metaestimators import if_has_delegate
    >>>
    >>> class A(object):
    ...     @if_has_delegate('d')
    ...     def func(self):
    ...         return True
    >>>
    >>> a = A()
    >>> # the delegate does not exist yet
    >>> assert not hasattr(a, 'func')
    >>> # inject the attribute
    >>> a.d = None
    >>> assert hasattr(a, 'func') and a.func()

    See Also
    --------
    :func:`pyramid.arima.ARIMA`

    Parameters
    ----------
    delegate : string, list of strings or tuple of strings
        Name of the sub-estimator that can be accessed as an attribute of the
        base object. If a list or a tuple of names are provided, the first
        sub-estimator that is an attribute of the base object will be used.
    """
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)

    return lambda fn: _IffHasDelegate(fn, delegate)
