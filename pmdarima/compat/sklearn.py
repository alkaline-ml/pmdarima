# -*- coding: utf-8 -*-
#
# Author: Charles Drotar <drotarcharles@gmail.com>
#
# Patch backend for sklearn

from pkg_resources import parse_version

import sklearn
from sklearn.utils.validation import check_is_fitted

__all__ = [
    'get_compatible_check_is_fitted',
    'safe_indexing',
]


def get_compatible_check_is_fitted(estimator, attributes=None):
    """Make the ``check_is_fitted`` function compatible across the
    sklearn 0.21 to 0.22+ boundary.


    Parameters
    ----------
    estimator : estimator instance,
        The estimator that will be checked to see if it is fitted.

    attributes : str, optional (default=None)
        The attributes to be passed into the ``check_is_fitted`` function.

    """

    # Use the version string from sklearn to determine how to use
    # the `sklearn.utils.validation import check_is_fitted` function
    # Determine if we should require a value for the attribute field
    is_no_attribute_needed = False
    if parse_version(sklearn.__version__) >= parse_version('0.22.0'):
        is_no_attribute_needed = True
        
    # If we need to pass in `attributes` but it is not a str type raise error.
    if not is_no_attribute_needed and not isinstance(attributes, str):
        raise TypeError("Parameter `attributes` can only accept str type")

    # If we don't need an attribute (i.e. sklearn.__version__ >= 0.22)
    # We ignore `attributes` to bypass warning
    if is_no_attribute_needed:
        return check_is_fitted(estimator=estimator)

    # Otherwise we use the attribute that was passed in regardless of version.
    return check_is_fitted(estimator=estimator, attributes=attributes)


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
