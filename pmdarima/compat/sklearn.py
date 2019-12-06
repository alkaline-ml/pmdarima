# -*- coding: utf-8 -*-
#
# Author: Charles Drotar <drotarcharles@gmail.com>
#
# Patch backend for sklearn

from __future__ import absolute_import

import sklearn
from sklearn.utils.validation import check_is_fitted

__all__ = [
    'get_compatible_check_is_fitted'
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

    # Split the version string from sklearn and use that determine how to use
    # the `sklearn.utils.validation import check_is_fitted` function
    semantic_version_split = [int(x) for x in sklearn.__version__.split(".")]
    major_version = semantic_version_split[0]
    minor_version = semantic_version_split[1]

    # Determine if we should require a value for the attribute field
    is_no_attribute_needed = False
    if major_version == 0 and minor_version > 21:
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
