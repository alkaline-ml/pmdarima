# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Tests for stationarity

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from statsmodels import api as sm
from abc import ABCMeta, abstractmethod
import numpy as np

from ..compat.numpy import DTYPE
from ..decorators import deprecated
from ..utils.array import c, diff, check_endog
from .approx import approx

# since the C import relies on the C code having been built with Cython,
# and since the platform might name the .so file something funky (like
# _arima.cpython-35m-darwin.so), import this absolutely and not relatively.
from ._arima import C_tseries_pp_sum

__all__ = [
    'ADFTest',
    'KPSSTest',
    'PPTest'
]


class _BaseStationarityTest(BaseEstimator, metaclass=ABCMeta):
    @staticmethod
    def _base_case(x):
        # if x is empty, return false so the other methods return False
        if (x is None) or (x.shape[0] == 0):
            return False
        return True

    @staticmethod
    def _embed(x, k):
        # lag the vector and put the lags into columns
        n = x.shape[0]
        if k > n:
            raise ValueError("k cannot exceed y dim")

        rows = [
            # so, if k=2, it'll be (x[1:n], x[:n-1])
            x[j:n - i] for i, j in enumerate(range(k - 1, -1, -1))
        ]
        return np.asarray(rows)
        # return np.array([x[1:], x[:m]])


class _DifferencingStationarityTest(_BaseStationarityTest, metaclass=ABCMeta):
    """Provides the base class for stationarity tests such as the
    Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller and the
    Phillips–Perron tests. These tests are used to determine whether a time
    series is stationary.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    @abstractmethod
    def should_diff(self, x):
        """Test whether the time series is stationary or it needs differencing.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        pval : float
            The computed P-value of the test.

        sig : bool
            Whether the P-value is significant at the ``alpha`` level.
            More directly, whether to difference the time series.
        """

    # TODO: REMOVE ME in v1.4.0
    @deprecated(use_instead="should_diff")
    def is_stationary(self, x):
        """Test whether the time series is stationary.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        pval : float
            The computed P-value of the test.

        sig : bool
            Whether the P-value is significant at the ``alpha`` level.
            More directly, whether to difference the time series.
        """
        return self.should_diff(x)


class KPSSTest(_DifferencingStationarityTest):
    """Conduct a KPSS test for stationarity.

    In econometrics, Kwiatkowski–Phillips–Schmidt–Shin (KPSS) tests are used
    for testing a null hypothesis that an observable time series is stationary
    around a deterministic trend (i.e. trend-stationary) against the
    alternative of a unit root.

    Parameters
    ----------
    alpha : float, optional (default=0.05)
        Level of the test

    null : str, optional (default='level')
        Whether to fit the linear model on the one vector, or an arange.
        If ``null`` is 'trend', a linear model is fit on an arange, if
        'level', it is fit on the one vector.

    lshort : bool, optional (default=True)
        Whether or not to truncate the ``l`` value in the C code.

    Notes
    -----
    This test is generally used indirectly via the
    :func:`pmdarima.arima.ndiffs` function, which computes the
    differencing term, ``d``.

    References
    ----------
    .. [1] R's tseries KPSS test source code: http://bit.ly/2eJP1IU
    """
    _valid = {'trend', 'null'}
    tablep = c(0.01, 0.025, 0.05, 0.10)

    def __init__(self, alpha=0.05, null='level', lshort=True):
        super(KPSSTest, self).__init__(alpha=alpha)

        self.null = null
        self.lshort = lshort

    def should_diff(self, x):
        """Test whether the time series is stationary or needs differencing.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        pval : float
            The computed P-value of the test.

        sig : bool
            Whether the P-value is significant at the ``alpha`` level.
            More directly, whether to difference the time series.
        """
        if not self._base_case(x):
            return np.nan, False

        # ensure vector
        x = check_endog(x, dtype=DTYPE, preserve_series=False)
        n = x.shape[0]

        # check on status of null
        null = self.null

        # fit a model on an arange to determine the residuals
        if null == 'trend':
            t = np.arange(n).reshape(n, 1)

            # these numbers came out of the R code.. I've found 0 doc for these
            table = c(0.216, 0.176, 0.146, 0.119)
        elif null == 'level':
            t = np.ones(n).reshape(n, 1)

            # these numbers came out of the R code.. I've found 0 doc for these
            table = c(0.739, 0.574, 0.463, 0.347)
        else:
            raise ValueError("null must be one of %r" % self._valid)

        # fit the model
        lm = LinearRegression().fit(t, x)
        e = x - lm.predict(t)  # residuals

        s = np.cumsum(e)
        eta = (s * s).sum() / (n**2)
        s2 = (e * e).sum() / n

        # scalar, denom = 10, 14
        # if self.lshort:
        #     scalar, denom = 3, 13
        # l_ = int(np.trunc(scalar * np.sqrt(n) / denom))
        if self.lshort:
            l_ = int(np.trunc(4 * (n / 100) ** 0.25))
        else:
            l_ = int(np.trunc(12 * (n / 100) ** 0.25))

        # compute the C subroutine
        s2 = C_tseries_pp_sum(e, n, l_, s2)
        stat = eta / s2

        # do approximation
        _, pval = approx(table, self.tablep, xout=stat, rule=2)

        # R does a test for rule=1, but we don't want to do that, because they
        # just do it to issue a warning in case the P-value is smaller/greater
        # than the printed value is.
        return pval[0], pval[0] < self.alpha


class ADFTest(_DifferencingStationarityTest):
    """Conduct an ADF test for stationarity.

    In statistics and econometrics, an augmented Dickey–Fuller test (ADF)
    tests the null hypothesis of a unit root is present in a time series
    sample. The alternative hypothesis is different depending on which version
    of the test is used, but is usually stationarity or trend-stationarity. It
    is an augmented version of the Dickey–Fuller test for a larger and more
    complicated set of time series models.

    Parameters
    ----------
    alpha : float, optional (default=0.05)
        Level of the test

    k : int, optional (default=None)
        The drift parameter. If ``k`` is None, it will be set to:
        ``np.trunc(np.power(x.shape[0] - 1, 1 / 3.0))``

    Notes
    -----
    This test is generally used indirectly via the
    :func:`pmdarima.arima.ndiffs` function, which computes the
    differencing term, ``d``.

    ADF test does not perform as close to the R code as do the KPSS and PP
    tests. This is due to the fact that is has to use statsmodels OLS
    regression for std err estimates rather than the more robust sklearn
    LinearRegression.

    References
    ----------
    .. [1] https://wikipedia.org/wiki/Augmented_Dickey–Fuller_test
    .. [2] R's tseries ADF source code: https://bit.ly/2EnvM5V
    """
    table = np.array([
        (-4.38, -3.95, -3.60, -3.24, -1.14, -0.80, -0.50, -0.15),
        (-4.15, -3.80, -3.50, -3.18, -1.19, -0.87, -0.58, -0.24),
        (-4.04, -3.73, -3.45, -3.15, -1.22, -0.90, -0.62, -0.28),
        (-3.99, -3.69, -3.43, -3.13, -1.23, -0.92, -0.64, -0.31),
        (-3.98, -3.68, -3.42, -3.13, -1.24, -0.93, -0.65, -0.32),
        (-3.96, -3.66, -3.41, -3.12, -1.25, -0.94, -0.66, -0.33)
    ])

    tablen = table.shape[1]
    tableT = c(25, 50, 100, 250, 500, 100000)
    tablep = c(0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99)

    def __init__(self, alpha=0.05, k=None):
        super(ADFTest, self).__init__(alpha=alpha)

        self.k = k
        if k is not None and k < 0:
            raise ValueError('k must be a positive integer (>= 0)')

    @staticmethod
    def _ols(x, y, z, k):
        n = y.shape[0]
        yt = z[:, 0]  # type: np.ndarray
        tt = np.arange(k - 1, n)

        # R does [k:n].. but that's 1-based indexing and inclusive on the tail
        xt1 = x[tt]

        # make tt inclusive again (it was used as a mask before)
        tt += 1

        # the array that will create the LM:
        _n = xt1.shape[0]  # row dim for predictors
        X = np.hstack([np.ones(_n).reshape((_n, 1)),
                       xt1.reshape((_n, 1)),
                       tt.reshape((_n, 1))])

        if k > 1:
            yt1 = z[:, 1:k]  # R had 2:k
            X = np.hstack([X, yt1])

        # fit the linear regression - this one is a bit strange in that we
        # are using OLS from statsmodels rather than LR from sklearn. This is
        # because we need the std errors, and sklearn does not have a way to
        # store them.
        return sm.OLS(yt, X, hasconst=True).fit(method='qr')

    @staticmethod
    def _ols_std_error(res):
        stderrs = res.bse  # this was a pain in the ARSE to locate
        return res.params[1] / stderrs[1]

    def should_diff(self, x):
        """Test whether the time series is stationary or needs differencing.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        pval : float
            The computed P-value of the test.

        sig : bool
            Whether the P-value is significant at the ``alpha`` level.
            More directly, whether to difference the time series.
        """
        if not self._base_case(x):
            return np.nan, False

        # ensure vector
        x = check_endog(x, dtype=DTYPE, preserve_series=False)

        # if k is none...
        k = self.k
        if k is None:
            k = np.trunc(np.power(x.shape[0] - 1, 1 / 3.0))

        # See [2] for the R source. This is L153 - L160
        k = int(k) + 1
        y = diff(x)  # diff(as.vector(x, mode='double'))
        n = y.shape[0]
        z = self._embed(y, k).T  # Same as R embed(x, k)

        # Compute ordinary least squares
        res = self._ols(x, y, z, k)
        STAT = self._ols_std_error(res)

        # In the past we assigned to the np memory view which is slower
        tableipl = np.array([
            approx(self.tableT, self.table[:, i], xout=n, rule=2)[1]  # xt,yt
            for i in range(self.tablen)])

        # make sure to do 1 - x...
        _, interpol = approx(tableipl, self.tablep, xout=STAT, rule=2)

        # Added in v1.1.0. Not sure whether it's likely we'll hit it, but it's
        # how R is warning the user...
        # if np.isnan(approx(tableipl, self.tablep, xout=STAT, rule=1)[1]):
        #     if interpol == self.tablep.min():
        #         warnings.warn("p-value is smaller than printed value")
        #     else:
        #         warnings.warn("p-value is larger than printed value")

        # pval = 1 - interpol[0]  # explosive
        pval = interpol[0]  # stationarity

        # in the R code, here is where the P value warning is tested again...
        # else if (test == "adf")
        #     suppressWarnings(dodiff < - tseries::adf.test(x)$p.value > alpha)
        return pval, pval > self.alpha  # > since not 1-pval


class PPTest(_DifferencingStationarityTest):
    """Conduct a PP test for stationarity.

    In statistics, the Phillips–Perron test (named after Peter C. B.
    Phillips and Pierre Perron) is a unit root test. It is used in time series
    analysis to test the null hypothesis that a time series is integrated of
    order 1. It builds on the Dickey–Fuller test of the null hypothesis
    ``p = 0``.

    Parameters
    ----------
    alpha : float, optional (default=0.05)
        Level of the test

    lshort : bool, optional (default=True)
        Whether or not to truncate the ``l`` value in the C code.

    Notes
    -----
    This test is generally used indirectly via the
    :func:`pmdarima.arima.ndiffs` function, which computes the
    differencing term, ``d``.

    The R code allows for two types of tests: 'Z(alpha)' and 'Z(t_alpha)'.
    Since sklearn does not allow extraction of std errors from the linear
    model fit, ``t_alpha`` is much more difficult to achieve, so we do not
    allow that variant.

    References
    ----------
    .. [1] R's tseries PP test source code: http://bit.ly/2wbzx6V
    """
    table = -np.array([
        (22.5, 25.7, 27.4, 28.4, 28.9, 29.5),
        (19.9, 22.4, 23.6, 24.4, 24.8, 25.1),
        (17.9, 19.8, 20.7, 21.3, 21.5, 21.8),
        (15.6, 16.8, 17.5, 18.0, 18.1, 18.3),
        (3.66, 3.71, 3.74, 3.75, 3.76, 3.77),
        (2.51, 2.60, 2.62, 2.64, 2.65, 2.66),
        (1.53, 1.66, 1.73, 1.78, 1.78, 1.79),
        (0.43, 0.65, 0.75, 0.82, 0.84, 0.87)
    ]).T

    tablen = table.shape[1]
    tableT = c(25, 50, 100, 250, 500, 100000).astype(DTYPE)
    tablep = c(0.01, 0.025, 0.05, 0.10, 0.90, 0.95, 0.975, 0.99)

    def __init__(self, alpha=0.05, lshort=True):
        super(PPTest, self).__init__(alpha=alpha)

        self.lshort = lshort

    def should_diff(self, x):
        """Test whether the time series is stationary or needs differencing.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        pval : float
            The computed P-value of the test.

        sig : bool
            Whether the P-value is significant at the ``alpha`` level.
            More directly, whether to difference the time series.
        """
        if not self._base_case(x):
            return np.nan, False

        # ensure vector
        x = check_endog(x, dtype=DTYPE, preserve_series=False)

        # embed the vector. This is some funkiness that goes on in the R
        # code... basically, make a matrix where the column (rows if not T)
        # are lagged windows of x
        z = self._embed(x, 2)  # Same as R t(embed(x, k))
        yt = z[0, :]
        yt1 = z[1, :]  # type: np.ndarray

        # fit a linear model to a predictor matrix
        n = yt.shape[0]
        tt = (np.arange(n) + 1) - (n / 2.0)
        X = np.array([np.ones(n), tt, yt1]).T
        res = LinearRegression().fit(X, yt)  # lm(yt ~ 1 + tt + yt1)
        coef = res.coef_

        # check for singularities - do we want to do this??? in the R code,
        # it happens. but the very same lm in the R code is rank 3, and here
        # it is rank 2. Should we just ignore?...
        # if res.rank_ < 3:
        #     raise ValueError('singularities in regression')

        u = yt - res.predict(X)  # residuals
        ssqru = (u * u).sum() / float(n)

        scalar = 12 if not self.lshort else 4
        l_ = int(np.trunc(scalar * np.power(n / 100.0, 0.25)))
        ssqrtl = C_tseries_pp_sum(u, n, l_, ssqru)

        # define trm vals
        n2 = n * n
        syt11n = (yt1 * (np.arange(n) + 1)).sum()  # sum(yt1*(1:n))
        trm1 = n2 * (n2 - 1) * (yt1 ** 2).sum() / 12.0

        # R code: # n*sum(yt1*(1:n))^2
        trm2 = n * (syt11n ** 2)

        # R code: n*(n+1)*sum(yt1*(1:n))*sum(yt1)
        trm3 = n * (n + 1) * syt11n * yt1.sum()
        trm4 = (n * (n + 1) * (2 * n + 1) * (yt1.sum() ** 2)) / 6.0
        dx = trm1 - trm2 + trm3 - trm4

        # if self.typ == 'alpha':
        alpha = coef[2]  # it's the last col...
        STAT = n * (alpha - 1) - (n ** 6) / (24.0 * dx) * (ssqrtl - ssqru)

        tableipl = np.array([
            approx(self.tableT, self.table[:, i], xout=n, rule=2)[1]
            for i in range(self.tablen)])

        # we don't do 1 - pval, so check for GREATER THAN
        _, interpol = approx(tableipl, self.tablep, xout=STAT, rule=2)
        pval = interpol[0]

        # in the R code, here is where the P value warning is tested again...
        return pval, pval > self.alpha
