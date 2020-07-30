# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Tests for seasonal differencing terms, and seasonal decomposition

from collections import namedtuple

import math
from sklearn.linear_model import LinearRegression

from scipy.linalg import svd
from statsmodels import api as sm
from statsmodels.tools import add_constant

from abc import ABCMeta, abstractmethod
from numpy.linalg import solve
import numpy as np

from .arima import _aicc
from ..compat.numpy import DTYPE
from .stationarity import _BaseStationarityTest
from ..utils.array import c, diff, check_endog

from ._arima import C_canova_hansen_sd_test

__all__ = [
    'CHTest',
    'decompose',
    'OCSBTest'
]


def decompose(x, type_, m, filter_=None):
    """
    Decompose the time series into trend, seasonal, and random components.

    Parameters
    ----------
    x : np.array, shape=(n_samples,)
        The time series of which the trend, seasonal, and noise/random
        components will be extracted.

    type_: str
        The type of decomposition that will be performed - 'multiplicative' or
        'additive'. We would use 'multiplicative' generally when we see an
        increasing trend. We use 'additive' when the trend is relatively
        stable over time.

    m: int
        The frequency in terms of number of observations. This behaves
        similarly to R's frequency for a time series (ts).

    filter_: np.array, optional (default=None)
        A filter by which the convolution will be performed.

    Returns
    -------
    decomposed_tuple : namedtuple
        A named tuple with ``x``, ``trend``, ``seasonal``, and ``random``
        components where ``x`` is the input signal, ``trend`` is the overall
        trend, ``seasonal`` is the seasonal component, and `random` is the
        noisy component. The input signal ``x`` can be mostly reconstructed by
        the other three components with a number of points missing equal to
        ``m``.

    Notes
    -----
    This function is generally used in conjunction with
    :func:`pmdarima.utils.visualization.decomposed_plot`,
    which plots the decomposed components. Also there is an example script in
    the ``examples`` folder of the repo and the ``Examples`` section of the
    docs as well.

     References
    ----------
    .. [1] Example of decompose using both multiplicative and additive types:
           https://anomaly.io/seasonal-trend-decomposition-in-r/index.html

    .. [2] R documentation for decompose:
           https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/decompose
    """  # noqa: E501

    multiplicative = "multiplicative"
    additive = "additive"
    is_m_odd = (m % 2 == 1)

    # Helper function to stay consistent and concise based on 'type_'
    def _decomposer_helper(a, b):
        if type_ == multiplicative:
            return a / b
        else:
            return a - b

    # Since R's ts class has a frequency as input I think this it acceptable
    # to ask the user for the frequency.
    try:
        assert isinstance(m, int) and m > 1
    except (ValueError, AssertionError):
        raise ValueError("'f' should be a positive integer")

    if filter_ is None:
        filter_ = np.ones((m,)) / m

    # We only accept the values in multiplicative or additive
    if type_ not in (multiplicative, additive):
        err_msg = "'type_' can only take values '{}' or '{}'"
        raise ValueError(err_msg.format(multiplicative, additive))

    # There needs to be at least 2 periods. This is due to the behavior of
    # convolutions and how they behave with respect to losing endpoints
    if (x.shape[0] / m) < 2:
        raise ValueError("time series has no or less than 2 periods")

    # Take half of m for the convolution / sma process.
    half_m = m // 2
    trend = np.convolve(x, filter_, mode='valid')

    if not is_m_odd:
        trend = trend[:-1]  # we remove the final index if m is even.

    # Remove the effect of the trend on the original signal and pad for reshape
    sma_xs = range(half_m, len(trend) + half_m)
    detrend = _decomposer_helper(x[sma_xs], trend)
    num_seasons = math.ceil((1.0 * trend.shape[0]) / m)
    pad_length = (num_seasons * m) - trend.shape[0]
    if pad_length > 0:
        buffer = pad_length * [np.nan]
        detrend = np.array(detrend.tolist() + buffer)

    # Determine the seasonal effect of the signal
    m_arr = np.reshape(detrend, (num_seasons, m))
    seasonal = np.nanmean(m_arr, axis=0).tolist()
    seasonal = np.array(seasonal[half_m:] + seasonal[:half_m])
    temp = seasonal
    for i in range(m_arr.shape[0]):
        seasonal = np.concatenate((seasonal, temp))
    if pad_length > 0:
        seasonal = seasonal[:-pad_length]
    if is_m_odd:
        seasonal = seasonal[:-1]

    # We buffer the trend and seasonal components so that they are the same
    # length as the other outputs. This counters the effects of losing data
    # by the convolution/sma
    buffer = [np.nan] * half_m
    trend = list(buffer + trend.tolist() + buffer)

    # Remove the trend and seasonal effects from the original signal to get
    # the random/noisy effects within the original signal.
    random = _decomposer_helper(_decomposer_helper(x, trend), seasonal)

    # Create a namedtuple so the output mirrors the output of the R function.
    decomposed = namedtuple('decomposed', 'x trend seasonal random')
    decomposed_tuple = decomposed(x, trend, seasonal, random)

    return decomposed_tuple


class _SeasonalStationarityTest(_BaseStationarityTest, metaclass=ABCMeta):
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
        x = check_endog(x, dtype=DTYPE)

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
    """Perform an OCSB test of seasonality.

    Compute the Osborn, Chui, Smith, and Birchenhall (OCSB) test for an input
    time series to determine whether it needs seasonal differencing. The
    regression equation may include lags of the dependent variable. When
    ``lag_method`` = "fixed", the lag order is fixed to ``max_lag``; otherwise,
    ``max_lag`` is the maximum number of lags considered in a lag selection
    procedure that minimizes the ``lag_method`` criterion, which can be
    "aic", "bic" or corrected AIC, "aicc".

    Critical values for the test are based on simulations, which have been
    smoothed over to produce critical values for all seasonal periods

    Parameters
    ----------
    m : int
        The seasonal differencing term. For monthly data, e.g., this would be
        12. For quarterly, 4, etc. For the OCSB test to work, ``m`` must
        exceed 1.

    lag_method : str, optional (default="aic")
        The lag method to use. One of ("fixed", "aic", "bic", "aicc"). The
        metric for assessing model performance after fitting a linear model.

    max_lag : int, optional (default=3)
        The maximum lag order to be considered by ``lag_method``.

    References
    ----------
    .. [1] Osborn DR, Chui APL, Smith J, and Birchenhall CR (1988)
           "Seasonality and the order of integration for consumption",
           Oxford Bulletin of Economics and Statistics 50(4):361-377.

    .. [2] R's forecast::OCSB test source code: https://bit.ly/2QYQHno
    """
    _ic_method_map = {
        "aic": lambda fit: fit.aic,
        "bic": lambda fit: fit.bic,

        # TODO: confirm False for add_constant, since the model fit contains
        #   . a constant term
        "aicc": lambda fit: _aicc(fit, fit.nobs, False)
    }

    def __init__(self, m, lag_method="aic", max_lag=3):
        super(OCSBTest, self).__init__(m=m)

        self.lag_method = lag_method
        self.max_lag = max_lag

    @staticmethod
    def _calc_ocsb_crit_val(m):
        """Compute the OCSB critical value"""
        # See:
        # https://github.com/robjhyndman/forecast/blob/
        # 8c6b63b1274b064c84d7514838b26dd0acb98aee/R/unitRoot.R#L409
        log_m = np.log(m)
        return -0.2937411 * \
            np.exp(-0.2850853 * (log_m - 0.7656451) + (-0.05983644) *
                   ((log_m - 0.7656451) ** 2)) - 1.652202

    @staticmethod
    def _do_lag(y, lag, omit_na=True):
        """Perform the TS lagging"""
        n = y.shape[0]
        if lag == 1:
            return y.reshape(n, 1)

        # Create a 2d array of dims (n + (lag - 1), lag). This looks cryptic..
        # If there are tons of lags, this may not be super efficient...
        out = np.ones((n + (lag - 1), lag)) * np.nan
        for i in range(lag):
            out[i:i + n, i] = y

        if omit_na:
            out = out[~np.isnan(out).any(axis=1)]
        return out

    @staticmethod
    def _gen_lags(y, max_lag, omit_na=True):
        """Create the lagged exogenous array used to fit the linear model"""
        if max_lag <= 0:
            return np.zeros(y.shape[0])

        # delegate down
        return OCSBTest._do_lag(y, max_lag, omit_na)

    @staticmethod
    def _fit_ocsb(x, m, lag, max_lag):
        """Fit the linear model used to compute the test statistic"""
        y_first_order_diff = diff(x, m)

        # if there are no more samples, we have to bail
        if y_first_order_diff.shape[0] == 0:
            raise ValueError(
                "There are no more samples after a first-order "
                "seasonal differencing. See http://alkaline-ml.com/pmdarima/"
                "seasonal-differencing-issues.html for a more in-depth "
                "explanation and potential work-arounds."
            )

        y = diff(y_first_order_diff)
        ylag = OCSBTest._gen_lags(y, lag)

        if max_lag > -1:
            # y = tail(y, -maxlag)
            y = y[max_lag:]

        # A constant term is added in the R code's lm formula. We do that in
        # the linear model's constructor
        mf = ylag[:y.shape[0]]
        ar_fit = sm.OLS(y, add_constant(mf)).fit(method='qr')

        # Create Z4
        z4_y = y_first_order_diff[lag:]  # new endog
        z4_lag = OCSBTest._gen_lags(y_first_order_diff, lag)[:z4_y.shape[0], :]
        z4_preds = ar_fit.predict(add_constant(z4_lag))  # preds
        z4 = z4_y - z4_preds  # test residuals

        # Create Z5. Looks odd because y and lag depend on each other and go
        # back and forth for two stages
        z5_y = diff(x)
        z5_lag = OCSBTest._gen_lags(z5_y, lag)
        z5_y = z5_y[lag:]
        z5_lag = z5_lag[:z5_y.shape[0], :]
        z5_preds = ar_fit.predict(add_constant(z5_lag))
        z5 = z5_y - z5_preds

        # Finally, fit a linear regression on mf with z4 & z5 features added
        data = np.hstack((
            mf,
            z4[:mf.shape[0]].reshape(-1, 1),
            z5[:mf.shape[0]].reshape(-1, 1)
        ))

        return sm.OLS(y, data).fit(method='qr')

    def _compute_test_statistic(self, x):
        m = self.m
        maxlag = self.max_lag
        method = self.lag_method

        # We might try multiple lags in this case
        crit_regression = None
        if maxlag > 0 and method != 'fixed':
            try:
                icfunc = self._ic_method_map[method]
            except KeyError:
                raise ValueError("'%s' is an invalid method. Must be one "
                                 "of ('aic', 'aicc', 'bic', 'fixed')")

            fits = []
            icvals = []
            for lag_term in range(1, maxlag + 1):  # 1 -> maxlag (incl)
                try:
                    fit = self._fit_ocsb(x, m, lag_term, maxlag)
                    fits.append(fit)
                    icvals.append(icfunc(fit))
                except np.linalg.LinAlgError:  # Singular matrix
                    icvals.append(np.nan)
                    fits.append(None)

            # If they're all NaN, raise
            if np.isnan(icvals).all():
                raise ValueError("All lag values up to 'maxlag' produced "
                                 "singular matrices. Consider using a longer "
                                 "series, a different lag term or a different "
                                 "test.")

            # Compute the information criterion vals
            best_index = int(np.nanargmin(icvals))
            maxlag = best_index - 1

            # Save this in case we can't compute a better one
            crit_regression = fits[best_index]

        # Compute the actual linear model used for determining the test stat
        try:
            regression = self._fit_ocsb(x, m, maxlag, maxlag)
        except np.linalg.LinAlgError:  # Singular matrix
            if crit_regression is not None:
                regression = crit_regression
            # Otherwise we have no solution to fall back on
            else:
                raise ValueError("Could not find a solution. Try a longer "
                                 "series, different lag term, or a different "
                                 "test.")

        # Get the coefficients for the z4 and z5 matrices
        tvals = regression.tvalues[-2:]  # len 2
        return tvals[-1]  # just z5, like R does it

    def estimate_seasonal_differencing_term(self, x):
        """Estimate the seasonal differencing term.

        Parameters
        ----------
        x : array-like, shape=(n_samples,)
            The time series vector.

        Returns
        -------
        D : int
            The seasonal differencing term. For different values of ``m``,
            the OCSB statistic is compared to an estimated critical value, and
            returns 1 if the computed statistic is greater than the critical
            value, or 0 if not.
        """
        if not self._base_case(x):
            return 0

        # ensure vector
        x = check_endog(x, dtype=DTYPE)

        # Get the critical value for m
        stat = self._compute_test_statistic(x)
        crit_val = self._calc_ocsb_crit_val(self.m)
        return int(stat > crit_val)
