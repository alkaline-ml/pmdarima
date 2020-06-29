# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Common ARIMA functions

from sklearn.utils.validation import column_or_1d
import numpy as np

import warnings

from .. import context_managers as ctx
from ..utils import get_callable
from ..utils.array import diff, check_endog
from ..compat.numpy import DTYPE
from . import stationarity as statest_lib
from . import seasonality as seatest_lib

__all__ = [
    'is_constant',
    'ndiffs',
    'nsdiffs'
]

VALID_TESTS = {
    'kpss': statest_lib.KPSSTest,
    'adf': statest_lib.ADFTest,
    'pp': statest_lib.PPTest
}

VALID_STESTS = {
    'ocsb': seatest_lib.OCSBTest,
    'ch': seatest_lib.CHTest
}


def is_constant(x):
    """Test ``x`` for constancy.

    Determine whether a vector is composed of all of the same elements
    and nothing else.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
        The time series vector.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> y = np.ones(3)
    >>> [is_constant(x), is_constant(y)]
    [False, True]
    """
    x = column_or_1d(x)  # type: np.ndarray
    return (x == x[0]).all()


def nsdiffs(x, m, max_D=2, test='ocsb', **kwargs):
    """Estimate the seasonal differencing term, ``D``.

    Perform a test of seasonality for different levels of ``D`` to
    estimate the number of seasonal differences required to make a given time
    series stationary. Will select the maximum value of ``D`` for which
    the time series is judged seasonally stationary by the statistical test.

    Parameters
    ----------
    x : array-like, shape=(n_samples, [n_features])
        The array to difference.

    m : int
        The number of seasonal periods (i.e., frequency of the
        time series)

    max_D : int, optional (default=2)
        Maximum number of seasonal differences allowed. Must
        be a positive integer. The estimated value of ``D`` will not
        exceed ``max_D``.

    test : str, optional (default='ocsb')
        Type of unit root test of seasonality to use in order
        to detect seasonal periodicity. Valid tests include ("ocsb", "ch").
        Note that the CHTest is very slow for large data.

    Returns
    -------
    D : int
        The estimated seasonal differencing term. This is the maximum value
        of ``D`` such that ``D <= max_D`` and the time series is judged
        seasonally stationary. If the time series is constant, will return 0.
    """
    if max_D <= 0:
        raise ValueError('max_D must be a positive integer')

    # get the test - this validates m internally
    testfunc = get_callable(test, VALID_STESTS)(m, **kwargs)\
        .estimate_seasonal_differencing_term
    x = check_endog(x, dtype=DTYPE, copy=False)

    if is_constant(x):
        return 0

    D = 0
    dodiff = testfunc(x)
    while dodiff == 1 and D < max_D:
        D += 1
        x = diff(x, lag=m)

        if is_constant(x):
            return D

        # Issue 351: if the differenced array is now shorter than the seasonal
        # periodicity, we need to bail out now.
        if len(x) < m:
            warnings.warn("Appropriate D value may not have been reached; "
                          "length of seasonally-differenced array (%i) is "
                          "shorter than m (%i). Using D=%i"
                          % (len(x), m, D))
            return D

        dodiff = testfunc(x)

    return D


def ndiffs(x, alpha=0.05, test='kpss', max_d=2, **kwargs):
    """Estimate ARIMA differencing term, ``d``.

    Perform a test of stationarity for different levels of ``d`` to
    estimate the number of differences required to make a given time
    series stationary. Will select the maximum value of ``d`` for which
    the time series is judged stationary by the statistical test.

    Parameters
    ----------
    x : array-like, shape=(n_samples, [n_features])
        The array (time series) to difference.

    alpha : float, optional (default=0.05)
        Level of the test. This is the value above below which the P-value
        will be deemed significant.

    test : str, optional (default='kpss')
        Type of unit root test of stationarity to use in order to
        test the stationarity of the time-series. One of ('kpss', 'adf', 'pp')

    max_d : int, optional (default=2)
        Maximum number of non-seasonal differences allowed. Must
        be a positive integer. The estimated value of ``d`` will not
        exceed ``max_d``.

    Returns
    -------
    d : int
        The estimated differencing term. This is the maximum value of ``d``
        such that ``d <= max_d`` and the time series is judged stationary.
        If the time series is constant, will return 0.

    References
    ----------
    .. [1] R's auto_arima ndiffs function
           https://github.com/robjhyndman/forecast/blob/19b0711e554524bf6435b7524517715658c07699/R/arima.R#L132  # noqa: E501
    """
    if max_d <= 0:
        raise ValueError('max_d must be a positive integer')

    # get the test
    testfunc = get_callable(test, VALID_TESTS)(alpha, **kwargs).should_diff
    x = check_endog(x, dtype=DTYPE, copy=False)

    # base case, if constant return 0
    d = 0
    if is_constant(x):
        return d

    with ctx.except_and_reraise(
            np.linalg.LinAlgError,
            raise_err=ValueError,
            raise_msg="Encountered exception in stationarity test (%r). "
                      "This can occur in seasonal settings when a large "
                      "enough `m` coupled with a large enough `D` difference "
                      "the training array into too few samples for OLS "
                      "(input contains %i samples). Try fitting on a larger "
                      "training size" % (test, len(x)),
    ):
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
