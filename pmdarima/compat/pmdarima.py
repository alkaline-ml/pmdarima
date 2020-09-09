# -*- coding: utf-8 -*-

import warnings


def get_X(X, **kwargs):
    """Get the exogenous array

    Gets the X array or ``exogenous`` array from ``kwargs``, warning for
    deprecation of old key-word arguments, and amends the ``kwargs`` array.
    """
    exog = kwargs.pop("exogenous", None)
    if X is not None and exog is not None:
        raise ValueError("Multiple values provided for both X and exogenous")

    if exog is not None:
        warnings.warn(
            "The `exogenous` key-word has been deprecated. Please "
            "use `X` instead. This will raise an error in future "
            "versions. For more information, see: "
            "http://alkaline-ml.com/pmdarima/develop/rfc/372-exog-to-x.html",
            DeprecationWarning
        )
        X = exog

    return X, kwargs
