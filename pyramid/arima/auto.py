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

from .utils import ndiffs, is_constant, nsdiffs
from ..utils.array import diff
from .arima import ARIMA

# for python 3
try:
    xrange
except NameError:
    xrange = range

# The DTYPE we'll use for everything here. Since there are
# lots of spots where we define the DTYPE in a numpy array,
# it's easier to define as a global for this module.
DTYPE = np.float64

__all__ = [
    'auto_arima'
]

# The valid information criteria
VALID_CRITERIA = {'aic', 'bic'}


def auto_arima(y, exogenous=None, start_p=2, d=None, start_q=2, max_p=5, max_d=2, max_q=5,
               start_P=1, D=None, start_Q=1, max_P=2, max_D=1, max_Q=2, max_order=None, m=1,
               seasonal=True, stationary=False, information_criterion='aic', alpha=0.05, test='kpss',
               seasonal_test='ch', n_jobs=1, start_params=None, trend='c', method=None, transparams=True,
               solver='lbfgs', maxiter=50, disp=0, callback=None, offset_test_args=None, seasonal_test_args=None,
               suppress_warnings=False, error_action='warn', **fit_args):
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
        The maximum value of ``d``, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than ``d``.

    max_q : int, optional (default=5)
        The maximum value of ``q``, inclusive. Must be a positive integer greater than
        ``start_q``.

    start_P : int, optional (default=1)
        The starting value of ``P``, the order of the auto-regressive portion
        of the seasonal model.

    D : int, optional (default=None)
        The order of the seasonal differencing. If None (by default_, the value
        will automatically be selected based on the results of the ``seasonal_test``.
        Must be a positive integer or None.

    start_Q : int, optional (default=1)
        The starting value of ``Q``, the order of the moving-average portion
        of the seasonal model.

    max_P : int, optional (default=2)
        The maximum value of ``P``, inclusive. Must be a positive integer greater
        than ``start_P``.

    max_D : int, optional (default=1)
        The maximum value of ``D``. Must be a positive integer greater than ``D``.

    max_Q : int, optional (default=2)
        The maximum value of ``Q``, inclusive. Must be a positive integer greater
        than ``start_Q``.

    max_order : int, optional (default=None)
        If the sum of ``p`` and ``q`` is >= ``max_order``, a model will *not* be
        fit with those parameters, but will progress to the next combination.
        Default is None, which means there are no constraints on maximum order.

    m : int, optional (default=1)
        The period for seasonal differencing. Typically, it is 4 for quarterly
        data, 12 for monthly data, or 1 for annual data. Default is 1.

    seasonal : bool, optional (default=True)
        Whether to fit a seasonal ARIMA.

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

    seasonal_test : str, optional (default='ch')
        This determines which seasonal unit root test is used.

    n_jobs : int, optional (default=1)
        The number of jobs to run if running in parallel.

    start_params : array-like, optional (default=None)
        Starting parameters for ``ARMA(p,q)``.  If None, the default is given
        by ``ARMA._fit_start_params``.

    transparams : bool, optional (default=True)
        Whehter or not to transform the parameters to ensure stationarity.
        Uses the transformation suggested in Jones (1980).  If False,
        no checking for stationarity or invertibility is done.

    method : str, one of {'css-mle','mle','css'}, optional (default=None)
        This is the loglikelihood to maximize.  If "css-mle", the
        conditional sum of squares likelihood is maximized and its values
        are used as starting values for the computation of the exact
        likelihood via the Kalman filter.  If "mle", the exact likelihood
        is maximized via the Kalman Filter.  If "css" the conditional sum
        of squares likelihood is maximized.  All three methods use
        `start_params` as starting parameters.  See above for more
        information. If fitting a seasonal ARIMA, the default is 'lbfgs'

    trend : str or iterable, optional (default='c')
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in ``numpy.poly1d``, where
        ``[1,1,0,1]`` would denote :math:`a + bt + ct^3`.

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
        parameter vector. This is only used in non-seasonal ARIMA models.

    offset_test_args : dict, optional (default=None)
        The args to pass to the constructor of the offset (``d``) test.

    seasonal_test_args : dict, optional (default=None)
        The args to pass to the constructor of the seasonal offset (``D``) test.

    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If ``suppress_warnings``
        is True, all of these warnings will be squelched.

    error_action : str, optional (default='warn')
        If unable to fit an ARIMA due to stationarity issues, whether to warn ('warn'),
        raise the ``ValueError`` ('raise') or ignore ('ignore').

    **fit_args : dict, optional (default=None)
        A dictionary of keyword arguments to pass to the :func:`ARIMA.fit` method.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    [2] https://github.com/robjhyndman/forecast/blob/19b0711e554524bf6435b7524517715658c07699/R/arima.R
    [3] https://www.rdocumentation.org/packages/forecast/versions/7.3/topics/auto.arima
    """
    # validate start/max points
    if any(_ < 0 for _ in (max_p, max_q, max_P, max_Q, start_p, start_q, start_P, start_Q)):
        raise ValueError('starting and max p, q, P & Q values must be positive integers (>= 0)')
    if max_p <= start_p or max_q <= start_q or max_P <= start_P or max_Q <= start_Q:
        raise ValueError('max p, q, P & Q must be less than their starting values')

    # validate max_order
    if max_order is None:
        max_order = np.inf
    elif max_order < 0:
        raise ValueError('max_order must be None or a positive integer (>= 0)')

    # validate d & D
    for _d, _max_d in ((d, max_d), (D, max_D)):
        if _max_d < 0:
            raise ValueError('max_d & max_D must be positive integers (>= 0)')
        if _d is not None:
            if _d < 0:
                raise ValueError('d & D must be None or a positive integer (>= 0)')
            if _d > _max_d:
                raise ValueError('if explicitly defined, d & D must be <= max_d & <= max_D, respectively')

    # check on m
    if m < 1:
        raise ValueError('m must be a positive integer (> 0)')

    # validate error action
    actions = {'warn', 'raise', 'ignore', None}
    if error_action not in actions:
        raise ValueError('error_action must be one of %r, but got %r' % (actions, error_action))

    # copy array
    y = check_array(y, ensure_2d=False, dtype=DTYPE, copy=True, force_all_finite=True)
    n_samples = y.shape[0]

    # check for constant data
    if is_constant(y):
        warnings.warn('Input time-series is completely constant; returning a (0, 0, 0) ARMA.')
        return _fit_arima(y, xreg=exogenous, order=(0, 0, 0), seasonal_order=None, start_params=start_params,
                          trend=trend, method=method, transparams=transparams, solver=solver,
                          maxiter=maxiter, disp=disp, callback=callback, fit_params=fit_args,
                          suppress_warnings=suppress_warnings, error_action=error_action)

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
    max_p = int(min(max_p, np.floor(n_samples / 3)))
    max_q = int(min(max_q, np.floor(n_samples / 3)))

    # if it's not seasonal, we can avoid multiple 'if not is None' comparisons
    # later by just using this shortcut (hack):
    if not seasonal:
        D = m = -1

    # choose the order of differencing
    xx = y.copy()
    if exogenous is not None:
        lm = LinearRegression().fit(exogenous, y)
        xx = y - lm.predict(exogenous)

    # is the TS stationary?
    if stationary:
        d = D = 0

    # now for seasonality
    if m == 1:
        D = max_P = max_Q = 0

    # m must be > 1 for nsdiffs
    elif D is None:  # we don't have a D yet and we need one (seasonal)
        seasonal_test_args = seasonal_test_args if seasonal_test_args is not None else dict()
        D = nsdiffs(xx, m=m, test=seasonal_test, max_D=max_D, **seasonal_test_args)
        if D > 0 and exogenous is not None:
            diffex = diff(exogenous, differences=D, lag=m)
            # check for constance on any column
            if np.apply_along_axis(is_constant, arr=diffex, axis=0).any():
                D -= 1

    # D might still be None if not seasonal. Py 3 will throw and error for that
    # unless we explicitly check for ``seasonal``
    if D > 0:
        dx = diff(xx, differences=D, lag=m)
    else:
        dx = xx

    # difference the exogenous matrix
    if exogenous is not None:
        if D > 0:
            diffex = diff(exogenous, differences=D, lag=m)
        else:
            diffex = exogenous

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
    if error_action == 'warn' and not suppress_warnings:
        if D >= 2:
            warnings.warn("Having more than one seasonal differences is not recommended. "
                          "Please consider using only one seasonal difference.")
        elif (D + d > 2 or d > 2):  # if D is -1, this will be off, so we include the OR
            warnings.warn("Having 3 or more differencing operations is not recommended. "
                          "Please consider reducing the total number of differences.")

    if d > 0:
        dx = diff(dx, differences=d, lag=1)

    # check for constance
    if is_constant(dx):
        if exogenous is None and not (D > 0 or d < 2):
            raise ValueError('data follow a simple polynomial and are not suitable for ARIMA modeling')

        # perfect regression
        ssn = None if not seasonal else (0, D, 0, m)
        return _fit_arima(y, xreg=exogenous, order=(0, d, 0), seasonal_order=ssn, start_params=start_params,
                          trend=trend, method=method, transparams=transparams, solver=solver,
                          maxiter=maxiter, disp=disp, callback=callback, fit_params=fit_args,
                          suppress_warnings=suppress_warnings, error_action=error_action)

    # seasonality issues
    if m > 1:
        if max_P > 0:
            max_p = min(max_p, m - 1)
        if max_Q > 0:
            max_q = min(max_q, m - 1)
        pass

    # generate the set of (p, q, P, Q) FIRST, since it is contingent on whether or not
    # the user is interested in a seasonal ARIMA result. This will reduce the search space
    # for non-seasonal ARIMA models.
    def generator():
        # loop p, q. Make sure to loop at +1 interval,
        # since max_{p|q} is inclusive.
        if seasonal:
            return (
                ((p, d, q), (P, D, Q, m))
                for p in xrange(start_p, max_p + 1)
                for q in xrange(start_q, max_q + 1)
                for P in xrange(start_P, max_P + 1)
                for Q in xrange(start_Q, max_Q + 1)
                if p + q + P + Q <= max_order
            )

        # otherwise it's not seasonal, and we don't need the seasonal pieces
        return (
            ((p, d, q), None)
            for p in xrange(start_p, max_p + 1)
            for q in xrange(start_q, max_q + 1)
            if p + q <= max_order
        )

    # get results in parallel
    gen = generator()  # the combos we need to fit
    all_res = Parallel(n_jobs=n_jobs)(
        delayed(_fit_arima)(y, xreg=exogenous, order=order, seasonal_order=seasonal_order,
                            start_params=start_params, trend=trend, method=method, transparams=transparams,
                            solver=solver, maxiter=maxiter, disp=disp, callback=callback,
                            fit_params=fit_args, suppress_warnings=suppress_warnings, error_action=error_action)
        for order, seasonal_order in gen)

    # filter the non-successful ones
    filtered = [m for m in all_res if m is not None]
    if not filtered:
        raise ValueError('No ARIMAs were successfully fit. It is likely your data is non-stationary. '
                         'Please induce stationarity or try a different range of model order params.')

    # sort by the criteria
    sorted_res = sorted(filtered, key=(lambda model: getattr(model, information_criterion)()), reverse=True)
    best = sorted_res[0]

    # remove all the cached .pmdpkl files...
    for model in sorted_res:
        model._clear_cached_state()

    return best


def _fit_arima(x, xreg, order, seasonal_order, start_params, trend, method, transparams,
               solver, maxiter, disp, callback, fit_params, suppress_warnings,
               error_action):
    try:
        return ARIMA(order=order, seasonal_order=seasonal_order, start_params=start_params,
                     trend=trend, method=method, transparams=transparams,
                     solver=solver, maxiter=maxiter, disp=disp,
                     callback=callback, suppress_warnings=suppress_warnings)\
            .fit(x, exogenous=xreg, **fit_params)

    # for non-stationarity errors, return None
    except ValueError as v:
        if error_action == 'warn':
            warnings.warn(_fmt_warning_str(order, seasonal_order))
        elif error_action == 'raise':
            raise v
        # otherwise it's 'ignore'
        return None


def _fmt_warning_str(order, seasonal_order):
    """This is just so we can test that the string will format even if we don't want the warnings in the tests"""
    return ('Unable to fit ARIMA for order=(%i, %i, %i)%s; data is likely non-stationary. '
            '(if you do not want to see these warnings, run with error_action="ignore")'
            % (order[0], order[1], order[2], '' if seasonal_order is None else ' seasonal_order=(%i, %i, %i, %i)'
               % (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_order[3])))
