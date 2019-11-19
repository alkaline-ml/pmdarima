# -*- coding: utf-8 -*-
#
# Taylor G Smith <taylor.smith@alkaline.ml>
#
# Wrapped functions

from __future__ import absolute_import

from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

from functools import wraps

__all__ = [
    'acf',
    'pacf'
]

# TODO: remove all explicit args/kwargs, making them *args, **kwargs


def inheritdoc(parent):
    """Inherit documentation from a parent

    Parameters
    ----------
    parent : callable
        The parent function or class that contains the sought-after
        docstring. If it doesn't have a docstring, this might behave
        in unexpected ways.
        
    Examples
    --------
    >>> def a(x=1):
    ...     '''This is documentation'''
    ...     return x
    ...
    >>> @inheritdoc(a)
    ... def b(x):
    ...     return 2 * a(x)
    ...
    >>> print(b.__doc__)
    This is documentation

    >>> print(b(2))
    4
    """
    def wrapper(func):
        # Assign the parent docstring to the child
        func.__doc__ = parent.__doc__

        @wraps(func)
        def caller(*args, **kwargs):
            return func(*args, **kwargs)
        return caller
    return wrapper


@inheritdoc(parent=sm_acf)
def acf(x, unbiased=False, nlags=40, qstat=False, fft=False,
        alpha=None, missing='none'):
    return sm_acf(x=x, unbiased=unbiased, nlags=nlags,
                  qstat=qstat, fft=fft, alpha=alpha,
                  missing=missing)


@inheritdoc(parent=sm_pacf)
def pacf(x, nlags=40, method='ywunbiased', alpha=None):
    return sm_pacf(x=x, nlags=nlags, method=method, alpha=alpha)
