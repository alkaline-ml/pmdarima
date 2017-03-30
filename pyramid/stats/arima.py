# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# ARIMAs

from __future__ import print_function, absolute_import, division
from sklearn.utils.validation import column_or_1d
from sklearn.base import BaseEstimator
from scipy.stats import boxcox
from .stationarity import check_test, is_constant
import numpy as np
import warnings

__all__ = [
    'AutoARIMA'
]


class _BaseARIMA(BaseEstimator):
    """
    """
    def __init__(self, xreg, allow_mean, allow_drift, lambda_val, bias_adjust):
        self.xreg = xreg
        self.allow_mean = allow_mean
        self.allow_drift = allow_drift
        self.lambda_val = lambda_val
        self.bias_adjust = bias_adjust


class ARIMA(_BaseARIMA):
    """
    """
    def __init__(self, p=0, d=0, q=0, P=0, D=0, Q=0, xreg=None, allow_mean=True,
                 allow_drift=False, lambda_val=None, bias_adjust=False, method='css-ml'):
        super(ARIMA, self).__init__(xreg=xreg, allow_mean=allow_mean,
                                    allow_drift=allow_drift,
                                    lambda_val=lambda_val,
                                    bias_adjust=bias_adjust)

        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.method = method

    def fit(self, x, y=None):
        # ensure that x is a 1d ts
        x = column_or_1d(x)
        origx = x[:]  # copy
        length = x.shape[0]

        # boxcox transform
        lam = self.lambda_val
        if lam is not None:
            x = boxcox(x=x, lmbda=lam)
        else:
            x, lam = boxcox(x=x, lmbda=None)

        # xreg
        if self.xreg is not None:
            #todo
            pass

        # check drift/seasonality
        drift = self.allow_drift
        if (self.d + self.D > 1) and drift:
            warnings.warn('No drift term fitted as the order of differences is 2 or more.')
            drift = False

        # todo: fit the actual model!

        # set the fit params
        self.lambda_ = lam

        return self


class AutoARIMA(_BaseARIMA):
    """An ARIMA, or autoregressive integrated moving average, is a generalization of an autoregressive
    moving average (ARMA) and is fitted to time-series data in an effort to forecast future points.
    ARIMA models can be especially efficacious in cases where data shows evidence of non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of interest is regressed on its own
    lagged (i.e., prior) values. The "MA" part indicates that the regression error is actually a linear
    combination of error terms whose values occurred contemporaneously and at various times in the past.
    The "I" (for "integrated") indicates that the data values have been replaced with the difference
    between their values and the previous values (and this differencing process may have been performed
    more than once). The purpose of each of these features is to make the model fit the data as well as possible.

    Non-seasonal ARIMA models are generally denoted ``ARIMA(p,d,q)`` where parameters ``p``, ``d``, and ``q`` are
    non-negative integers, ``p`` is the order (number of time lags) of the autoregressive model, ``d`` is the degree
    of differencing (the number of times the data have had past values subtracted), and ``q`` is the order of the
    moving-average model. Seasonal ARIMA models are usually denoted ``ARIMA(p,d,q)(P,D,Q)m``, where ``m`` refers
    to the number of periods in each season, and the uppercase ``P``, ``D``, ``Q`` refer to the autoregressive,
    differencing, and moving average terms for the seasonal part of the ARIMA model.

    When two out of the three terms are zeros, the model may be referred to based on the non-zero parameter,
    dropping "AR", "I" or "MA" from the acronym describing the model. For example, ``ARIMA(1,0,0)`` is ``AR(1)``,
    ``ARIMA(0,1,0)`` is ``I(1)``, and ``ARIMA(0,0,1)`` is ``MA(1)``. [1]

    Parameters
    ----------
    start_p : int, optional (default=2)
        The starting value of ``p``, the order (or number of time lags) of the autoregressive ("AR") model.

    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value will automatically be
        selected based on the results of the ``test``.

    start_q : int, optional (default=2)
        The starting value of ``q``, the order of the moving-average ("MA") model.

    start_P : int, optional (default=1)
        The starting value of ``P`` in stepwise procedure.

    D : int, optional (default=None)
        The order of seasonal differencing. If None (by default), the value will
        automatically be selected based on the results of the ``test``.

    start_Q : int, optional (default=1)
        The starting value of ``Q`` in stepwise procedure.

    max_p : int, optional (default=5)
        The maximum value of ``p``.

    max_d : int, optional (default=2)
        The maximum value of ``d``, or the maximum number of non-seasonal differences.

    max_q : int, optional (default=5)
        The maximum value of ``q``.

    max_P :

    max_D :

    max_Q :

    max_order :

    stationary :

    seasonal :

    information_criterion :

    stepwise :

    trace :

    approximation :

    truncate :

    xreg :

    test :

    seasonal_test :

    allow_drift :

    allow_mean :

    lambda_val :

    bias_adjust :

    n_jobs :


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    """

    def __init__(self, start_p=2, d=None, start_q=2, start_P=1, D=None, start_Q=1, max_p=5, max_d=2, max_q=5,
                 max_P=2, max_D=1, max_Q=2, max_order=5, stationary=False, seasonal=True, information_criterion='aic',
                 stepwise=True, trace=False, approximation=None, truncate=None, xreg=None, test='kpss',
                 seasonal_test='ocsb', allow_drift=True, allow_mean=True, lambda_val=None, bias_adjust=False,
                 n_jobs=1):

        super(AutoARIMA, self).__init__(xreg=xreg, allow_mean=allow_mean,
                                        allow_drift=allow_drift,
                                        lambda_val=lambda_val,
                                        bias_adjust=bias_adjust)

        # p,d,q are the most important terms for an ARIMA
        self.start_p = start_p
        self.d = d
        self.start_q = start_q

        # seasonal terms
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q

        # ceilings on the lags, degree and order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q

        # ceilings on seasonal terms
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q

        # other fun stuff
        self.max_order = max_order
        self.stationary = stationary
        self.seasonal = seasonal
        self.information_criterion = information_criterion
        self.stepwise = stepwise
        self.trace = trace
        self.approximation = approximation
        self.truncate = truncate
        self.test = test
        self.seasonal_test = seasonal_test
        self.n_jobs = n_jobs

    def fit(self, x, y=None):
        # only non-stepwise parallel implemented
        if self.stepwise and self.n_jobs > 1:
            warnings.warn("Parallel computation is only implemented when stepwise=False. "
                          "The model will be fit in serial.")
            self.n_jobs = 1

        # ensure that x is a 1d ts
        x = column_or_1d(x)
        length = x.shape[0]

        # is constant? return an arima
        if is_constant(x):
            # fit an arima.... make sure to set ``fixed`` in ARIMA fit
            fit = ARIMA(p=0, d=0, q=0, P=0, D=0, Q=0, allow_mean=self.allow_mean, xreg=self.xreg,
                        allow_drift=self.allow_drift, lambda_val=self.lambda_val,
                        bias_adjust=self.bias_adjust).fit(x)

            # todo: set internal attrs...

            return self

        # AICC doesn't work for very small samples
        ic = self.information_criterion
        if length <= 3:
            ic = 'aic'

        # get the functions
        test = check_test(self.test)
