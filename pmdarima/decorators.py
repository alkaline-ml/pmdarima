# -*- coding: utf-8 -*-

import functools
import warnings

__all__ = ['deprecated']


def deprecated(use_instead, notes=None):
    """Mark functions as deprecated.

    This decorator will result in a warning being emitted when the decorated
    function is used.

    Parameters
    ----------
    use_instead : str
        The name of the function to use instead.

    notes : str, optional (default=None)
        Additional notes to add to the warning message.
    """
    if notes is None:
        notes = ""
    else:
        notes = " " + notes

    def wrapped_func(func):
        @functools.wraps(func)
        def _inner(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # un-filter
            msg = ("{0} is deprecated and will be removed in a future "
                   "release of pmdarima. Use {1} instead.{2}"
                   .format(func.__name__, use_instead, notes))

            warnings.warn(
                msg,
                category=DeprecationWarning,
                stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # re-filter
            return func(*args, **kwargs)
        return _inner
    return wrapped_func
