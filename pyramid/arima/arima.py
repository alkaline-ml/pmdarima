# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# ARIMA

from __future__ import print_function, absolute_import, division

import numpy as np
from sklearn.base import BaseEstimator
from statsmodels.tsa.arima_model import ARIMA as arima

__all__ = [
    'ARIMA'
]


class ARIMA(BaseEstimator):
    """An ARIMA, or autoregressive integrated moving average, is a generalization of an autoregressive
    moving average (ARMA) and is fitted to time-series data in an effort to forecast future points.
    ARIMA models can be especially efficacious in cases where data shows evidence of non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of interest is regressed on its own
    lagged (i.e., prior observed) values. The "MA" part indicates that the regression error is actually a linear
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
    endogenous : array-like
        The endogenous variable.

    order : iterable or array-like, shape=(3,)
        The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters
        to use. ``p`` is the order (number of time lags) of the auto-regressive model, and is a non-
        negative integer. ``d`` is the degree of differencing (the number of times the data have
        had past values subtracted), and is a non-negative integer. ``q`` is the order of the moving-
        average model, and is a non-negative integer.

    exogenous : array-like, optional (default=None)
        An optional array of exogenous variables. This should not include a constant or trend.

    dates : array-like of datetime, optional
        An array-like object of datetime objects. If a pandas object is given for ``endogenous`` or
        ``exogenous``, it is assumed to have a DateIndex.

    freq : str, optional (default=None)
        The frequency of the time-series. A Pandas offset or 'B', 'D', 'W', 'M',
        'A', or 'Q'. This is optional if dates are given.

    start_params : array-like, optional (default=None)
        Starting parameters for ``ARMA(p,q)``.  If None, the default is given
        by ``ARMA._fit_start_params``.

    transparams : bool, optional (default=True)
        Whehter or not to transform the parameters to ensure stationarity.
        Uses the transformation suggested in Jones (1980).  If False,
        no checking for stationarity or invertibility is done.

    method : str, one of {'css-mle','mle','css'}, optional (default='css-mle')
        This is the loglikelihood to maximize.  If "css-mle", the
        conditional sum of squares likelihood is maximized and its values
        are used as starting values for the computation of the exact
        likelihood via the Kalman filter.  If "mle", the exact likelihood
        is maximized via the Kalman Filter.  If "css" the conditional sum
        of squares likelihood is maximized.  All three methods use
        `start_params` as starting parameters.  See above for more
        information.

    trend : str {'c','nc'}, optional (default='c')
        Whether to include a constant or not.  'c' includes constant,
        'nc' no constant.

    solver : str or None, optional (default='lbfgs')
        Solver to be used.  The default is 'lbfgs' (limited memory
        Broyden-Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs',
        'newton' (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' -
        (conjugate gradient), 'ncg' (non-conjugate gradient), and
        'powell'. By default, the limited memory BFGS uses m=12 to
        approximate the Hessian, projected gradient tolerance of 1e-8 and
        factr = 1e2. You can change these by using kwargs.

    maxiter : int, optional (default=50)
        The maximum number of function evaluations. Default is 50.

    full_output : bool, optional (default=1)
        If True, all output from solver will be available in
        the result object's mle_return vals attribute.  Output is dependent
        on the solver.  See Notes for more information.

    disp : int, optional (default=5)
        If True, convergence information is printed.  For the default
        'lbfgs' ``solver``, disp controls the frequency of the output during
        the iterations. disp < 0 means no output in this case.

    callback : callable, optional (default=None)
        Called after each iteration as callback(xk) where xk is the current
        parameter vector.

    start_ar_lags : int, optional (default=None)
        Parameter for fitting start_params. When fitting start_params,
        residuals are obtained from an AR fit, then an ARMA(p,q) model is
        fit via OLS using these residuals. If start_ar_lags is None, fit
        an AR process according to best BIC. If start_ar_lags is not None,
        fits an AR process with a lag length equal to start_ar_lags.
            See ARMA._fit_start_params_hr for more information.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    """
    def __init__(self, endogenous, order, exogenous=None, dates=None, freq=None, missing='none',
                 start_params=None, trend='c', method="css-mle", transparams=True, solver='lbfgs',
                 maxiter=50, full_output=1, disp=5, callback=None, start_ar_lags=None):
        super(ARIMA, self).__init__()

        self.endogenous = endogenous
        self.order = order
        self.exogenous = exogenous
        self.dates = dates
        self.freq = freq
        self.missing = missing
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.transparams = transparams
        self.solver = solver
        self.maxiter = maxiter
        self.full_output = full_output
        self.disp = disp
        self.callback = callback
        self.start_ar_lags = start_ar_lags

    def fit(self, x):
        pass


# The DTYPE we'll use for everything here. Since there are
# lots of spots where we define the DTYPE in a numpy array,
# it's easier to define as a global for this module.
DTYPE = np.float32
