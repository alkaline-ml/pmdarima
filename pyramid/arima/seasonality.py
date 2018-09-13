# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Tests for seasonal differencing terms

from __future__ import absolute_import, division

from sklearn.utils.validation import column_or_1d, check_array
from sklearn.linear_model import LinearRegression
from sklearn.externals import six

from scipy.linalg import svd
from abc import ABCMeta, abstractmethod

from numpy.linalg import solve
import numpy as np

from ..utils.array import c
from .stationarity import _BaseStationarityTest
from ..compat.numpy import DTYPE
from ._arima import C_canova_hansen_sd_test

__all__ = [
    'CHTest'
]


class _SeasonalStationarityTest(six.with_metaclass(ABCMeta,
                                                   _BaseStationarityTest)):
    """Provides the base class for seasonal differencing tests such as the
    Canova-Hansen test and the Osborn-Chui-Smith-Birchenhall tests. These tests
    are used to determine the seasonal differencing term for a time-series.
    """
    def __init__(self, m):
        self.m = m
        if m < 2:
            raise ValueError('m must be > 1')

    @abstractmethod
    def estimate_seasonal_differencing_term(self, x):
        """Estimate the seasonal differencing term.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.
        """


class CHTest(_SeasonalStationarityTest):
    """Conduct a CH test for seasonality.

    The Canova-Hansen test for seasonal differences. Canova and Hansen
    (1995) proposed a test statistic for the null hypothesis that the seasonal
    pattern is stable. The test statistic can be formulated in terms of
    seasonal dummies or seasonal cycles. The former allows us to identify
    seasons (e.g. months or quarters) that are not stable, while the latter
    tests the stability of seasonal cycles (e.g. cycles of period 2 and 4
    quarters in quarterly data). [1]

    Parameters
    ----------
    m : int
        The seasonal differencing term. For monthly data, e.g., this would be
        12. For quarterly, 4, etc. For the Canova-Hansen test to work,
        ``m`` must exceed 1.

    Notes
    -----
    This test is generally not used directly, but in conjunction with
    :func:`pyramid.arima.nsdiffs`, which directly estimates the number
    of seasonal differences.

    References
    ----------
    .. [1] Testing for seasonal stability using the Canova
           and Hansen test statisic: http://bit.ly/2wKkrZo
    """
    def __init__(self, m):
        super(CHTest, self).__init__(m=m)

    @staticmethod
    def _sd_test(wts, s):
        # assume no NaN values since called internally
        # also assume s > 1 since called internally
        n = wts.shape[0]

        # no use checking, because this is an internal method
        # if n <= s:  raise ValueError('too few samples (%i<=%i)' % (n, s))
        frec = np.ones(int((s + 1) / 2), dtype=np.int)
        ltrunc = int(np.round(s * ((n / 100.0) ** 0.25)))
        R1 = CHTest._seas_dummy(wts, s)

        # fit model, get residuals
        lmch = LinearRegression().fit(R1, wts)
        residuals = wts - lmch.predict(R1)

        # translated R code:
        # multiply the residuals by the column vectors
        # Fhataux = Fhat.copy()
        # for i in range(Fhat.shape[1]):  # for (i in 1:(s-1))
        #     Fhataux[:, i] = R1[:, i] * residuals

        # more efficient numpy:
        Fhataux = (R1.T * residuals).T

        # translated R code
        # matrix row cumsums
        # Fhat = np.ones((n, s - 1)) * np.nan
        # for i in range(n):
        #    for n in range(Fhataux.shape[1]):
        #         Fhat[i, n] = Fhataux[:i, n].sum()

        # more efficient numpy:
        Fhat = Fhataux.cumsum(axis=0)
        Ne = Fhataux.shape[0]

        # As of v0.9.1, use the C_canova_hansen_sd_test function to compute
        # Omnw, Omfhat, A, frecob. This avoids the overhead of multiple calls
        # to C functions
        A, AtOmfhatA = C_canova_hansen_sd_test(ltrunc, Ne, Fhataux, frec, s)

        # UPDATE 01/04/2018 - we can get away without computing u, v
        # (this is also MUCH MUCH faster!!!)
        sv = svd(AtOmfhatA, compute_uv=False)  # type: np.ndarray

        # From R:
        # double.eps: the smallest positive floating-point number ‘x’ such that
        # ‘1 + x != 1’.  It equals ‘double.base ^ ulp.digits’ if either
        # ‘double.base’ is 2 or ‘double.rounding’ is 0; otherwise, it
        # is ‘(double.base ^ double.ulp.digits) / 2’.  Normally
        # ‘2.220446e-16’.
        # Numpy's float64 has an eps of 2.2204460492503131e-16
        if sv.min() < np.finfo(sv.dtype).eps:  # machine min eps
            return 0

        # solve against the identity matrix, then produce
        # a nasty mess of dot products... this is the (horrendous) R code:
        # (1/N^2) * sum(diag(solve(tmp) %*% t(A) %*% t(Fhat) %*% Fhat %*% A))
        # https://github.com/robjhyndman/forecast/blob/master/R/arima.R#L321
        solved = solve(AtOmfhatA, np.identity(AtOmfhatA.shape[0]))
        return (1.0 / n ** 2) * solved.dot(A.T).dot(
            Fhat.T).dot(Fhat).dot(A).diagonal().sum()

    @staticmethod
    def _seas_dummy(x, m):
        # set up seasonal dummies using fourier series
        n = x.shape[0]

        # assume m > 1 since this function called internally...
        assert m > 1, 'This function is called internally and ' \
                      'should not encounter this issue'

        tt = np.arange(n) + 1
        fmat = np.ones((n, 2 * m)) * np.nan
        pi = np.pi
        for i in range(1, m + 1):  # for(i in 1:m)
            # subtract one, unlike the R code. in the R code, this sets
            # columns 2, 4, 6, etc... here it sets 1, 3, 5
            # fmat[,2*i] <- sin(2*pi*i*tt/m)
            fmat[:, (2 * i) - 1] = np.sin(2 * pi * i * tt / m)

            # in the R code, this sets columns 1, 3, 5, etc. here it
            # sets 0, 2, 4, etc.
            # fmat[,2*(i-1)+1] <- cos(2*pi*i*tt/m)
            fmat[:, 2 * (i - 1)] = np.cos(2 * pi * i * tt / m)

        return fmat[:, :m - 1]

    def estimate_seasonal_differencing_term(self, x):
        """Estimate the seasonal differencing term.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        D : int
            The seasonal differencing term. The CH test defines a set of
            critical values::

                (0.4617146, 0.7479655, 1.0007818,
                 1.2375350, 1.4625240, 1.6920200,
                 1.9043096, 2.1169602, 2.3268562,
                 2.5406922, 2.7391007)

            For different values of ``m``, the CH statistic is compared
            to a different critical value, and returns 1 if the computed
            statistic is greater than the critical value, or 0 if not.
        """
        if not self._base_case(x):
            return 0

        # ensure vector
        x = column_or_1d(check_array(
            x, ensure_2d=False, dtype=DTYPE,
            force_all_finite=True))  # type: np.ndarray

        n = x.shape[0]
        m = int(self.m)

        if n < 2 * m + 5:
            return 0

        chstat = self._sd_test(x, m)
        crit_vals = c(0.4617146, 0.7479655, 1.0007818,
                      1.2375350, 1.4625240, 1.6920200,
                      1.9043096, 2.1169602, 2.3268562,
                      2.5406922, 2.7391007)

        if m <= 12:
            return int(chstat > crit_vals[m - 2])  # R does m - 1...
        if m == 24:
            return int(chstat > 5.098624)
        if m == 52:
            return int(chstat > 10.341416)
        if m == 365:
            return int(chstat > 65.44445)

        return int(chstat > 0.269 * (m ** 0.928))
