# -*- coding: utf-8 -*-
#
# Taylor G Smith <taylor.smith@alkaline.ml>
#
# Wrapped functions
from functools import wraps
from pkg_resources import parse_version
import warnings

from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
import statsmodels

from pmdarima.compat.statsmodels import _use_sm13

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
def acf(x, unbiased=None, nlags=None, qstat=False, fft=None,
        alpha=None, missing='none', adjusted=False):
    kwargs = {
        "x": x,
        "nlags": nlags,
        "qstat": qstat,
        "fft": fft,
        "alpha": alpha,
        "missing": missing,
    }

    # Handle kwarg deprecation in statsmodels 0.13.0
    if _use_sm13():
        if unbiased is not None:
            warnings.warn(
                "The `unbiased` key-word has been deprecated in "
                "statsmodels >= 0.13.0. Please use `adjusted` instead.",
                DeprecationWarning
            )
            kwargs["adjusted"] = unbiased
        else:
            kwargs["adjusted"] = adjusted

        kwargs["fft"] = fft or True
    else:
        kwargs["nlags"] = nlags or 40  # Becomes `None` in 0.13.0
        kwargs["fft"] = fft or False  # Becomes `True` in 0.13.0

        if unbiased is not None:
            kwargs["unbiased"] = unbiased
        else:
            kwargs["unbiased"] = adjusted

    return sm_acf(**kwargs)


@inheritdoc(parent=sm_pacf)
def pacf(x, nlags=None, method='ywadjusted', alpha=None):
    # Handle kwarg deprecation in statsmodels 0.13.0
    if _use_sm13():
        if "unbiased" in method:
            warnings.warn(
                "The `*unbiased` methods have been deprecated in "
                "statsmodels >= 0.13.0. Please use `*adjusted` instead.",
                DeprecationWarning
            )
            method = method.replace("unbiased", "adjusted")
        elif method in ("ydu", "ywu", "ldu"):
            warnings.warn(
                "The `ydu`, `ywu`, and `ldu` methods have been deprecated in "
                "statsmodels >= 0.13.0. Please use `yda`, `ywa`, and `lda` "
                "instead.",
                DeprecationWarning
            )
            method = method.replace("u", "a")
    else:
        # Have to do the opposite of the above since we support statsmodels
        # 0.11.0. 0.12.0 supports both `unbiased` and `adjusted`, but 0.11.0
        # doesn't, so this works for anything < 0.13.0
        if "adjusted" in method:
            method = method.replace("adjusted", "unbiased")
        elif method in ("yda", "ywa", "lda"):
            method = method.replace("a", "u")

        nlags = nlags or 40  # Becomes `None` in 0.13.0

    return sm_pacf(x=x, nlags=nlags, method=method, alpha=alpha)
