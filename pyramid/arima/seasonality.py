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

# since the C import relies on the C code having been built with Cython,
# and since the platform might name the .so file something funky (like
# _arima.cpython-35m-darwin.so), import this absolutely and not relatively.
from pyramid.arima._arima import C_pop_A

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
    """The Canova-Hansen test for seasonal differences. Canova and Hansen
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
        12. For quarterly, 4. For annual, 1, etc. For the Canova-Hansen test
        to work, ``m`` must exceed 1.

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

        # wnw <- 1 - seq(1, ltrunc, 1)/(ltrunc + 1)
        wnw = 1. - (np.arange(ltrunc) + 1.) / (ltrunc + 1.)
        Ne = Fhataux.shape[0]
        Omnw = 0

        # original R code:
        # for (k in 1:ltrunc)
        #     Omnw <- Omnw + (t(Fhataux)[, (k + 1):Ne] %*%
        #         Fhataux[1:(Ne - k), ]) * wnw[k]

        # translated R code:
        for k in range(ltrunc):
            Omnw = Omnw + (Fhataux.T[:, k + 1:Ne].dot(
                Fhataux[:(Ne - (k + 1)), :])) * wnw[k]

        # Omfhat <- (crossprod(Fhataux) + Omnw + t(Omnw))/Ne
        Omfhat = (Fhataux.T.dot(Fhataux) + Omnw + Omnw.T) / Ne
        sq = np.arange(0, s - 1, 2)
        frecob = np.zeros(s - 1).astype('int64')

        # I hate looping like this, but it seems like overkill to
        # write a C function for something that's otherwise so trivial...
        for i, v in enumerate(frec):
            if v == 1 and i == int(s / 2) - 1:
                frecob[sq[i]] = 1
            if v == 1 and i < int(s / 2) - 1:
                frecob[sq[i]] = frecob[sq[i] + 1] = 1

        # call the C sequence in place
        # C_frec(frec, frecob, sq, s)
        a = (frecob == 1).sum()

        # populate the A matrix
        A = np.zeros((s - 1, a)).astype('int64')
        C_pop_A(A, frecob)

        tmp = A.T.dot(Omfhat).dot(A)
        _, sv, _ = svd(tmp)
        if sv.min() < 2.220446e-16:  # machine min eps
            return 0

        # solve against the identity matrix, then produce
        # a nasty mess of dot products...
        solved = solve(tmp, np.identity(tmp.shape[0]))
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
