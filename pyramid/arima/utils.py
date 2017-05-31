# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Common ARIMA functions

from __future__ import absolute_import
from sklearn.utils.validation import check_array, column_or_1d
import numpy as np

from ..utils import get_callable
from ..utils.array import diff
from .stationarity import KPSSTest, ADFTest, PPTest

__all__ = [
    'frequency',
    'get_callable',
    'is_constant',
    'ndiffs'
]

VALID_TESTS = {
    'kpss': KPSSTest,
    'adf': ADFTest,
    'pp': PPTest
}


def frequency(x):
    # todo: should make a ts class?
    return 1


def is_constant(x):
    """Determine whether a vector is composed of all of the
    same elements and nothing else.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
        The time series vector.
    """
    return (x == x[0]).all()


def ndiffs(x, alpha=0.05, test='kpss', max_d=2, **kwargs):
    """Functions to estimate the number of differences required to
    make a given time series stationary.

    Parameters
    ----------
    x : array-like, shape=(n_samples, [n_features])
        The array to difference.

    alpha : float, optional (default=0.05)
        Level of the test

    test : str, optional (default='kpss')
        Type of unit root test to use in order to detect
        stationarity.

    max_d : int, optional (default=2)
        Maximum number of non-seasonal differences allowed. Must
        be a positive integer.
    """
    if max_d <= 0:
        raise ValueError('max_d must be a positive integer')

    # get the test
    testfunc = get_callable(test, VALID_TESTS)(alpha, **kwargs).is_stationary
    x = column_or_1d(check_array(x, ensure_2d=False, force_all_finite=True))
    d = 0

    # base case
    if is_constant(x):
        return d

    # get initial diff
    pval, dodiff = testfunc(x)

    # if initially NaN, return 0
    if np.isnan(pval):
        return 0  # (d is zero, but this is more explicit to the reader)

    # Begin loop.
    while dodiff and d < max_d:
        d += 1

        # do differencing
        x = diff(x)
        if is_constant(x):
            return d

        # get new result
        pval, dodiff = testfunc(x)

        # if it's NaN now, take the last non-null one
        if np.isnan(pval):
            return d - 1

    # when d >= max_d
    return d
