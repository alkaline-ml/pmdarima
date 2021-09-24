# -*- coding: utf-8 -*-
#
# Author: Charles Drotar <drotarcharles@gmail.com>
#
# Patch backend for sklearn

import sklearn
from sklearn.exceptions import NotFittedError

__all__ = [
    'check_is_fitted',
    'safe_indexing',
]


def check_is_fitted(estimator, attributes):
    """Ensure the model has been fitted

    Typically called at the beginning of an operation on a model that requires
    having been fit. Raises a ``NotFittedError`` if the model has not been
    fit.

    This is an adaptation of scikit-learn's ``check_is_fitted``, which has been
    changed recently in a way that is no longer compatible with our package.

    Parameters
    ----------
    estimator : estimator instance,
        The estimator that will be checked to see if it is fitted.

    attributes : str or iterable
        The attributes to check for
    """
    if isinstance(attributes, str):
        attributes = [attributes]
    if not hasattr(attributes, "__iter__"):
        raise TypeError("attributes must be a string or iterable")
    for attr in attributes:
        if hasattr(estimator, attr):
            return
    raise NotFittedError("Model has not been fit!")


def safe_indexing(X, indices):
    """Slice an array or dataframe. This is deprecated in sklearn"""
    if hasattr(X, 'iloc'):
        return X.iloc[indices]
    # numpy:
    # TODO: this does not currently support axis 1
    if hasattr(X, 'ndim') and X.ndim == 2:
        return X[indices, :]
    # list or 1d array
    return X[indices]


def _estimator_has(attr):
    def check(self):
        return hasattr(self.estimator, attr)

    return check


def if_delegate_has_method(attr):
    if sklearn.__version__ <= "0.24.2":
        from sklearn.utils.metaestimators import if_delegate_has_method
        return if_delegate_has_method(attr)
    else:
        from sklearn.utils.metaestimators import available_if
        return available_if(_estimator_has(attr))
