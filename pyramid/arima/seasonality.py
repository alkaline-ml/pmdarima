# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Tests for seasonal differencing terms

from __future__ import absolute_import, division
from statsmodels.regression.linear_model import OLS
from sklearn.utils.validation import column_or_1d, check_array
from sklearn.linear_model import LinearRegression
from sklearn.externals import six
from scipy.linalg import svd
from abc import ABCMeta, abstractmethod
from numpy.linalg import solve
import numpy as np

from ..utils.array import c, diff
from .arima import ARIMA
from .stationarity import _BaseStationarityTest

# since the C import relies on the C code having been built with Cython,
# and since pyramid never plans to use setuptools for the 'develop' option,
# make this an absolute import.
from pyramid.arima._arima import C_pop_A


class _SeasonalStationarityTest(six.with_metaclass(ABCMeta, _BaseStationarityTest)):
    """Provides the base class for seasonal differencing tests such as the
    Canova-Hansen test and the Osborn-Chui-Smith-Birchenhall tests. These tests are
    used to determine the seasonal differencing term for a time-series.
    """
    def __init__(self, m):
        self.m = m
        if m < 2:
            raise ValueError('m must be > 1')

    @abstractmethod
    def estimate_seasonal_differencing_term(self, x):
        """Estimate the seasonal differencing term (duh)"""


class CHTest(_SeasonalStationarityTest):
    """The Canova-Hansen test for seasonal differences.

    Parameters
    ----------
    m : int
        The seasonal differencing term.
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
        #     Omnw <- Omnw + (t(Fhataux)[, (k + 1):Ne] %*% Fhataux[1:(Ne - k), ]) * wnw[k]

        # translated R code:
        for k in range(ltrunc):
            Omnw = Omnw + (Fhataux.T[:, k + 1:Ne].dot(Fhataux[:(Ne - (k + 1)), :])) * wnw[k]

        # Omfhat <- (crossprod(Fhataux) + Omnw + t(Omnw))/Ne
        Omfhat = (Fhataux.T.dot(Fhataux) + Omnw + Omnw.T) / Ne
        sq = np.arange(0, s - 1, 2)
        frecob = np.zeros(s - 1, dtype=np.int)

        # I hate looping like this, but it seems like overkill to write a C function
        # for something that's otherwise so trivial...
        for i, v in enumerate(frec):
            if v == 1 and i == int(s / 2) - 1:
                frecob[sq[i]] = 1
            if v == 1 and i < int(s / 2) - 1:
                frecob[sq[i]] = frecob[sq[i] + 1] = 1

        # call the C sequence in place
        # C_frec(frec, frecob, sq, s)
        a = (frecob == 1).sum()

        # populate the A matrix
        A = np.zeros((s - 1, a), dtype=np.int)
        C_pop_A(A, frecob)

        tmp = A.T.dot(Omfhat).dot(A)
        _, s, _ = svd(tmp)
        if s.min() < 2.220446e-16:  # machine min eps
            return 0

        # solve against the identity matrix, then produce a nasty mess of dot products...
        solved = solve(tmp, np.identity(tmp.shape[0]))
        return (1.0 / n ** 2) * solved.dot(A.T).dot(Fhat.T).dot(Fhat).dot(A).diagonal().sum()


    @staticmethod
    def _seas_dummy(x, m):
        # set up seasonal dummies using fourier series
        n = x.shape[0]

        # assume m > 1 since this function called internally...
        assert m > 1, 'This function is called internally and should not encounter this issue'

        tt = np.arange(n) + 1
        fmat = np.ones((n, 2 * m)) * np.nan
        pi = np.pi
        for i in range(1, m + 1):  # for(i in 1:m)
            # subtract one, unlike the R code. in the R code, this sets
            # columns 2, 4, 6, etc... here it sets 1, 3, 5
            fmat[:, (2 * i) - 1] = np.sin(2 * pi * i * tt / m)  # fmat[,2*i] <- sin(2*pi*i*tt/m)

            # in the R code, this sets columns 1, 3, 5, etc. here it
            # sets 0, 2, 4, etc.
            fmat[:, 2 * (i - 1)] = np.cos(2 * pi * i * tt / m)  # fmat[,2*(i-1)+1] <- cos(2*pi*i*tt/m)

        return fmat[:, :m - 1]

    def estimate_seasonal_differencing_term(self, x):
        if not self._base_case(x):
            return 0

        # ensure vector
        x = column_or_1d(check_array(x, ensure_2d=False, dtype=np.float64, force_all_finite=True))
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


class OCSBTest(_SeasonalStationarityTest):
    def __init__(self, m):
        super(OCSBTest, self).__init__(m=m)

    @staticmethod
    def _calc_ocsb_crit_val(m):
        log_m = np.log(m)
        return -0.2937411 * np.exp(-0.2850853 * (log_m - 0.7656451) + (-0.05983644) *
                                   ((log_m - 0.7656451) ** 2)) - 1.652202

    def estimate_seasonal_differencing_term(self, x):
        if not self._base_case(x):
            return 0

        # ensure vector
        x = column_or_1d(check_array(x, ensure_2d=False, dtype=np.float64, force_all_finite=True))
        n = x.shape[0]
        m = int(self.m)

        if n < 2 * m + 5:
            return 0

        # Compute (1-B)(1-B^m)y_t
        seas_diff_series = diff(x, lag=m, differences=1)
        diff_series = diff(seas_diff_series, lag=1, differences=1)

        # Compute (1-B^m)y_{t-1} (we know it's > len 1)
        y_one = diff(x[1:], lag=m, differences=1)

        # Compute (1-B)y_{t-m}
        y_two = diff(x[m:], lag=1, differences=1)

        # make all series of the same length and matching time periods
        y_one = y_one[m:-1]
        y_two = y_two[1:-m]
        diff_series = diff_series[m + 1:]
        contingent_series = diff_series.copy()

        # this is hideous, but it's seriously cleaner than the way the R code does it...
        # https://github.com/robjhyndman/forecast/blob/30308a4e314ff29338291462e81bf68ff0c5f86d/R/newarima2.R#L839
        xreg = np.vstack([y_one, y_two]).T  # now they're columns

        orders = [
            ((3, 0, 0), (1, 0, 0, m)),
            ((3, 0, 0), (0, 0, 0, m)),
            ((2, 0, 0), (0, 0, 0, m)),
            ((1, 0, 0), (0, 0, 0, m))
        ]

        successful_arima = False
        for order, seasonal_order in orders:
            try:
                regression = ARIMA(order=order, seasonal_order=seasonal_order).fit(y=diff_series, exogenous=xreg)
                successful_arima = True
                break
            except ValueError:
                pass

        # if we were unable to fit a successful ARIMA, do this part:
        if not successful_arima:
            try:
                regression = OLS(contingent_series, exog=xreg - 1).fit()
            except:
                raise ValueError('The OCSB regression model cannot be estimated')

            residuals = contingent_series - regression.predict(xreg - 1)
            meanratio = np.abs(residuals).mean() / np.abs(contingent_series).mean()

            if np.isnan(meanratio) or meanratio < 1e-10:
                return 0

            # Proceed to do OCSB test on the linear model.
            y_two_t = regression.tvalues[1]  # the t-value for the y_two feature
            if not np.isfinite(y_two_t):
                return 1
            return int(y_two_t >= self._calc_ocsb_crit_val(m))

        else:
            # do the OCSB test on the ARIMA model
            # todo - under construction...
            # https://github.com/robjhyndman/forecast/blob/30308a4e314ff29338291462e81bf68ff0c5f86d/R/newarima2.R#L886

            return 0
