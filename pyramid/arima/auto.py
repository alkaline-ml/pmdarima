# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Automatically find optimal parameters for an ARIMA

from __future__ import absolute_import
from sklearn.utils.validation import check_array
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings

from .utils import ndiffs, is_constant
from ..utils.array import diff
from .arima import ARIMA

# The DTYPE we'll use for everything here. Since there are
# lots of spots where we define the DTYPE in a numpy array,
# it's easier to define as a global for this module.
DTYPE = np.float64

__all__ = [
    'auto_arima'
]

# The valid information criteria
VALID_CRITERIA = {'aic', 'bic'}


def auto_arima(y, exogenous=None, start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5, max_order=None,
               stationary=False, information_criterion='aic', alpha=0.05, test='kpss', n_jobs=1,
               start_params=None, trend='c', method="css-mle", transparams=True, solver='lbfgs',
               maxiter=50, disp=0, callback=None, offset_test_args=None, suppress_warnings=False, **fit_args):
    """The ``AutoARIMA`` function seeks to identify the most optimal parameters for an ``ARIMA`` model,
    and returns a fitted ARIMA model. This function is based on the commonly-used R function,
    `forecase::auto.arima``[3].

    Parameters
    ----------
    y : array-like, shape=(n_samples,)
        The time-series to which to fit an ``ARIMA`` estimator.

    exogenous : array-like, shape=[n_samples, n_features], optional (default=None)
        An optional array of exogenous variables. This should not
        include a constant or trend.

    start_p : int, optional (default=2)
        The starting value of ``p``, the order (or number of time lags)
        of the auto-regressive ("AR") model. Must be a positive integer.

    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value
        will automatically be selected based on the results of the ``test``.
        Must be a positive integer or None.

    start_q : int, optional (default=2)
        The starting value of ``q``, the order of the moving-average
        ("MA") model. Must be a positive integer.

    max_p : int, optional (default=5)
        The maximum value of ``p``, inclusive. Must be a positive integer greater
        than ``start_p``.

    max_d : int, optional (default=2)
        The maximum value of ``d``, inclusive, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than ``d``.

    max_q : int, optional (default=5)
        The maximum value of ``q``, inclusive. Must be a positive integer greater than
        ``start_q``.

    max_order : int, optional (default=None)
        If the sum of ``p`` and ``q`` is >= ``max_order``, a model will *not* be
        fit with those parameters, but will progress to the next combination.
        Default is None, which means there are no constraints on maximum order.

    stationary : bool, optional (default=False)
        Whether the time-series is stationary and ``d`` should be set to zero.

    information_criterion : str, optional (default='aic')
        The information criterion used to select the best ARIMA model. One of
        ``pyramid.arima.auto_arima.VALID_CRITERIA``, ('aic', 'bic'). Note that if
        n_samples <= 3, AIC will be used.

    alpha : float, optional (default=0.05)
        Level of the test for testing significance.

    test : str, optional (default='kpss')
        Type of unit root test to use in order to detect
        stationarity.

    n_jobs : int, optional (default=1)
        The number of jobs to run if running in parallel.

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

    disp : int, optional (default=0)
        If True, convergence information is printed.  For the default
        'lbfgs' ``solver``, disp controls the frequency of the output during
        the iterations. disp < 0 means no output in this case.

    callback : callable, optional (default=None)
        Called after each iteration as callback(xk) where xk is the current
        parameter vector.

    offset_test_args : dict, optional (default=None)
        The args to pass to the constructor of the offset (``d``) test.

    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If ``suppress_warnings``
        is True, all of these warnings will be squelched.

    **fit_args : dict, optional (default=None)
        A dictionary of keyword arguments to pass to the :func:`ARIMA.fit` method.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    [2] https://github.com/robjhyndman/forecast/blob/19b0711e554524bf6435b7524517715658c07699/R/arima.R
    [3] https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima
    """
    # validate start/max points
    if any(_ < 0 for _ in (max_p, max_q, start_p, start_q)):
        raise ValueError('starting and max p & q values must be positive integers (>= 0)')
    if max_p <= start_p or max_q <= start_q:
        raise ValueError('max p & q must be less than their starting values')

    # validate max_order
    if max_order is None:
        max_order = np.inf
    elif max_order < 0:
        raise ValueError('max_order must be None or a positive integer (>= 0)')

    # validate d
    if max_d < 0:
        raise ValueError('max_d must be a positive integer (>= 0)')
    if d is not None:
        if d < 0:
            raise ValueError('d must be None or a positive integer (>= 0)')
        if d > max_d:
            raise ValueError('if explicitly defined, d must be <= max_d')

    # copy array
    y = check_array(y, ensure_2d=False, dtype=DTYPE, copy=True, force_all_finite=True)
    n_samples = y.shape[0]

    # check for constant data
    if is_constant(y):
        warnings.warn('Input time-series is completely constant; returning a (0, 0, 0) ARMA.')
        return ARIMA(order=(0, 0, 0), start_params=start_params, trend=trend, method=method,
                     transparams=transparams, solver=solver, maxiter=maxiter, disp=disp,
                     callback=callback, suppress_warnings=suppress_warnings)\
            .fit(y, exogenous, **fit_args)

    # test ic, and use AIC if n <= 3
    if information_criterion not in VALID_CRITERIA:
        raise ValueError('auto_arima not defined for information_criteria=%s. '
                         'Valid information criteria include: %r'
                         % (information_criterion, VALID_CRITERIA))

    if n_samples <= 3:
        if information_criterion != 'aic':
            warnings.warn('n_samples (%i) <= 3 necessitates using AIC' % n_samples)
        information_criterion = 'aic'

    # adjust max p, q -- R code:
    # max.p <- min(max.p, floor(serieslength/3))
    # max.q <- min(max.q, floor(serieslength/3))
    max_p = min(max_p, np.floor(n_samples / 3))
    max_q = min(max_q, np.floor(n_samples / 3))

    # choose the order of differencing
    xx = y.copy()
    if exogenous is not None:
        lm = LinearRegression().fit(exogenous, y)
        xx = y - lm.predict(exogenous)

    # is the TS stationary?
    if stationary:
        # D = 0
        d = 0

    # now for seasonality. Since we cannot use P, D, Q, m in the ARIMA (dernit statsmodels),
    # we have to treat everything as non-seasonal. But we'll go thru the logical process
    # as if seasonality were a factor, in case they ever build it in.
    m = 1  # this will be removed if m is a parameter
    if m == 1:
        # D = max_P = max_Q = 0
        pass  # if seasonality ever works..
    # elif D is None:  # we don't have a D yet
    #     D = nsdiffs(xx, m=m, test=seasonal_test, max_D=max_D)
    #     todo: check on exogenous not null after differencing
    #     pass

    # if D > 0:
    #     dx = diff(xx, differences=D, lag=m)
    # else:
    dx = xx

    # difference the exogenous matrix
    diffex = exogenous
    if exogenous is not None:
        # if D > 0
        #     diffex = diff(exogenous, differences=D, lag=m)
        pass

    # determine/set the order of differencing by estimating the number of
    # orders it would take in order to make the TS stationary.
    if d is None:
        offset_test_args = offset_test_args if offset_test_args is not None else dict()
        d = ndiffs(dx, test=test, alpha=alpha, max_d=max_d, **offset_test_args)

        if d > 0 and exogenous is not None:
            diffex = diff(diffex, differences=d, lag=1)

            # if any columns are constant, subtract one order of differencing
            if np.apply_along_axis(is_constant, arr=diffex, axis=0).any():
                d -= 1

    # check differences (do we want to warn?...)
    # if D >= 2:
    #     warnings.warn("Having more than one seasonal differences is not recommended. "
    #                   "Please consider using only one seasonal difference.")
    # elif (D + d > 2):
    #     warnings.warn("Having 3 or more differencing operations is not recommended. "
    #                   "Please consider reducing the total number of differences.")

    if d > 0:
        dx = diff(dx, differences=d, lag=1)

    # check for constance
    if is_constant(dx):
        if exogenous is None:
            # if D > 0
            #     fit = ARIMA(order=(0, d, 0), seasonal=(0, D, 0), ...)
            # elif ...
            if d < 2:
                fit = ARIMA(order=(0, d, 0), start_params=start_params, trend=trend, method=method,
                            transparams=transparams, solver=solver, maxiter=maxiter, disp=disp,
                            callback=callback, suppress_warnings=suppress_warnings)\
                    .fit(y, exogenous, **fit_args)
            else:
                raise ValueError('data follow a simple polynomial and are not suitable for ARIMA modeling')
        else:  # perfect regression
            # if D > 0
            #     fit = ARIMA(order=(0, d, 0), seasonal=(0, D, 0), ...)
            # else:
            fit = ARIMA(order=(0, d, 0), start_params=start_params, trend=trend, method=method,
                        transparams=transparams, solver=solver, maxiter=maxiter, disp=disp,
                        callback=callback, suppress_warnings=suppress_warnings)\
                .fit(y, exogenous, **fit_args)

        return fit

    # seasonality issues
    if m > 1:  # won't be until seasonal ARIMA implemented
        # if max_P > 0:
        #     max_p = min(max_p, m - 1)
        # if max_Q > 0:
        #     max_q = min(max_q, m - 1)
        pass

    # get results in parallel
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(_fit_arima)(y, xreg=exogenous, order=(p, d, q), start_params=start_params,
                            trend=trend, method=method, transparams=transparams,
                            solver=solver, maxiter=maxiter, disp=disp, callback=callback,
                            fit_params=fit_args, suppress_warnings=suppress_warnings)

        # loop p, q. Make sure to loop at +1 interval,
        # since max_{p|q} is inclusive. Also, if we ever
        # add seasonality here, this will grow a bit in the loop
        for p in range(start_p, max_p + 1)
        for q in range(start_q, max_q + 1)
        if p + q <= max_order)

    # filter the non-successful ones
    filtered = [m for m in all_res if m is not None]
    if not filtered:
        raise ValueError('No ARIMAs were successfully fit. It is likely your data is non-stationary. '
                         'Please induce stationarity or try a different range of model order params.')

    # sort by the criteria
    sorted_res = sorted(filtered, key=(lambda model: getattr(model, information_criterion)()), reverse=True)
    return sorted_res[0]


def _fit_arima(x, xreg, order, start_params, trend, method, transparams,
               solver, maxiter, disp, callback, fit_params, suppress_warnings):
    try:
        return ARIMA(order=order, start_params=start_params,
                     trend=trend, method=method, transparams=transparams,
                     solver=solver, maxiter=maxiter, disp=disp,
                     callback=callback, suppress_warnings=suppress_warnings)\
            .fit(x, exogenous=xreg, **fit_params)

    # for non-stationarity errors, return None
    except ValueError:
        return None
