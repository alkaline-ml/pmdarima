# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Tests for seasonal differencing terms

from __future__ import absolute_import, division

from sklearn.externals import six
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import column_or_1d, check_array

from scipy.linalg import svd
from statsmodels import api as sm

from abc import ABCMeta, abstractmethod
from numpy.linalg import solve
import numpy as np

from ..compat.numpy import DTYPE
from .stationarity import _BaseStationarityTest
from ..utils.array import c, diff

from ._arima import C_canova_hansen_sd_test

__all__ = [
    'CHTest',
    'OCSBTest'
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
    :func:`pmdarima.arima.nsdiffs`, which directly estimates the number
    of seasonal differences.

    References
    ----------
    .. [1] Testing for seasonal stability using the Canova
           and Hansen test statisic: http://bit.ly/2wKkrZo

    .. [2] R source code for CH test:
           https://github.com/robjhyndman/forecast/blob/master/R/arima.R#L148
    """
    crit_vals = c(0.4617146, 0.7479655, 1.0007818,
                  1.2375350, 1.4625240, 1.6920200,
                  1.9043096, 2.1169602, 2.3268562,
                  2.5406922, 2.7391007)

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
        lmch = LinearRegression(normalize=True).fit(R1, wts)
        # lmch = sm.OLS(wts, R1).fit(method='qr')
        residuals = wts - lmch.predict(R1)

        # translated R code:
        # multiply the residuals by the column vectors
        # Fhataux = Fhat.copy()
        # for i in range(Fhat.shape[1]):  # for (i in 1:(s-1))
        #     Fhataux[:, i] = R1[:, i] * residuals

        # more efficient numpy:
        Fhataux = (R1.T * residuals).T.astype(np.float64)

        # translated R code
        # matrix row cumsums
        # Fhat = np.ones((n, s - 1)) * np.nan
        # for i in range(n):
        #    for n in range(Fhataux.shape[1]):
        #         Fhat[i, n] = Fhataux[:i, n].sum()

        # more efficient numpy:
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
        Fhat = Fhataux.cumsum(axis=0)
        solved = solve(AtOmfhatA, np.identity(AtOmfhatA.shape[0]))
        return (1.0 / n ** 2) * solved.dot(A.T).dot(
            Fhat.T).dot(Fhat).dot(A).diagonal().sum()

    @staticmethod
    def _seas_dummy(x, m):
        # Here is the R code:
        # (https://github.com/robjhyndman/forecast/blob/master/R/arima.R#L132)
        #
        # SeasDummy <- function(x) {
        #   n <- length(x)
        #   m <- frequency(x)
        #   if (m == 1) {
        #     stop("Non-seasonal data")
        #   }
        #   tt <- 1:n
        #   fmat <- matrix(NA, nrow = n, ncol = 2 * m)
        #   for (i in 1:m) {
        #     fmat[, 2 * i] <- sin(2 * pi * i * tt / m)
        #     fmat[, 2 * (i - 1) + 1] <- cos(2 * pi * i * tt / m)
        #   }
        #   return(fmat[, 1:(m - 1)])
        # }
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

        if m <= 12:
            return int(chstat > self.crit_vals[m - 2])  # R does m - 1...
        if m == 24:
            return int(chstat > 5.098624)
        if m == 52:
            return int(chstat > 10.341416)
        if m == 365:
            return int(chstat > 65.44445)

        return int(chstat > 0.269 * (m ** 0.928))


class OCSBTest(_SeasonalStationarityTest):
    """

    Parameters
    ----------
    m : int
        The seasonal differencing term. For monthly data, e.g., this would be
        12. For quarterly, 4, etc. For the OCSB test to work, ``m`` must
        exceed 1.

    lag_method : str, optional (default="aic")
        The lag method to use. One of ("fixed", "aic", "bic", "aicc"). The
        metric for assessing model performance after fitting a linear model.
    """
    def __init__(self, m, lag_method="aic", max_lag=3):
        super(OCSBTest, self).__init__(m=m)

        self.lag_method = lag_method
        self.max_lag = max_lag

    @staticmethod
    def _calc_ocsb_crit_val(m):
        log_m = np.log(m)
        return -0.2937411 * \
               np.exp(-0.2850853 * (log_m - 0.7656451) + (-0.05983644) *
                      ((log_m - 0.7656451) ** 2)) - 1.652202

    @staticmethod
    def _do_lag(y, lag):
        n = y.shape[0]
        if lag == 1:
            return y.reshape(n, 1)

        # Create a 2d array of dims (n + (lag - 1), lag). This looks cryptic..
        # If there are tons of lags, this may not be super efficient...
        out = np.ones((n + (lag - 1), lag)) * np.nan
        for i in range(lag):
            out[i:i + n, i] = y
        return out

    @staticmethod
    def _gen_lags(y, max_lag):
        if max_lag == 0:
            return np.zeros(y.shape[0])

        # delegate down
        return OCSBTest._do_lag(y, max_lag)

    @staticmethod
    def _fit_ocsb(x, m, lag, max_lag):
        y_first_order_diff = diff(x, m)
        y = diff(y_first_order_diff)
        ylag = OCSBTest._gen_lags(y, lag)

        if max_lag > 0:
            # y = tail(y, -maxlag)
            y = y[max_lag:]

        # Make a 2-feature matrix, but first trim out any rows with NaN in
        # the lag array
        # ylag_finite = ylag[~np.isnan(ylag).any(axis=1)]
        ylag_finite = ylag[lag:-lag, :]

        # The constant term is not part of the R code, but is used in the R lm
        mf = np.hstack((ylag_finite,
                        np.zeros((y.shape[0], 1))))

        # Fit the first linear model
        ar_fit = LinearRegression(fit_intercept=False).fit(mf, y)

        # Create Z4
        z4_lag = OCSBTest._gen_lags(y_first_order_diff, lag)
        z4lag_finite = z4_lag[~np.isnan(z4_lag).any(axis=1)]
        Z4_frame = np.hstack((y_first_order_diff,
                              z4lag_finite[:y_first_order_diff.shape[0]]))

        # TODO: finish this

    def estimate_seasonal_differencing_term(self, x):

        if not self._base_case(x):
            return 0

        m = int(self.m)
        if m == 1:
            raise ValueError("m must exceed 1")

        # ensure vector
        x = column_or_1d(
            check_array(x, ensure_2d=False, dtype=np.float64,
                        force_all_finite=True))  # type: np.ndarray

        n = x.shape[0]
        if n < 2 * m + 5:
            return 0

        # TODO: validate method

        maxlag = self.max_lag
        method = self.lag_method
        if maxlag > 0:
            if method != 'fixed':
                # TODO:
                pass

        regression = self._fit_ocsb(x, m, maxlag, maxlag)

        # Get the coefficients for the z4 and z5 matrices
        # TODO:
        stat = np.nan

        # Get the critical value for m
        crit_val = self._calc_ocsb_crit_val(m)
        return int(stat > crit_val)
