# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Automatically find optimal parameters for an ARIMA

from __future__ import absolute_import

from sklearn.utils.validation import check_array, column_or_1d
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from numpy.linalg import LinAlgError
import numpy as np

import warnings
import time

from .utils import ndiffs, is_constant, nsdiffs
from ..utils import diff, is_iterable
from .arima import ARIMA
from .warnings import ModelFitWarning

# for python 3 compat
from ..compat.python import xrange
from ..compat.numpy import DTYPE

__all__ = [
    'auto_arima'
]

# The valid information criteria
VALID_CRITERIA = {'aic', 'aicc', 'bic', 'hqic', 'oob'}


def auto_arima(y, exogenous=None, start_p=2, d=None, start_q=2, max_p=5,
               max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
               max_D=1, max_Q=2, max_order=10, m=1, seasonal=True,
               stationary=False, information_criterion='aic', alpha=0.05,
               test='kpss', seasonal_test='ch', stepwise=True, n_jobs=1,
               start_params=None, trend='c', method=None, transparams=True,
               solver='lbfgs', maxiter=50, disp=0, callback=None,
               offset_test_args=None, seasonal_test_args=None,
               suppress_warnings=False, error_action='warn', trace=False,
               random=False, random_state=None, n_fits=10,
               return_valid_fits=False, out_of_sample_size=0, scoring='mse',
               scoring_args=None, **fit_args):
    """Automatically discover the optimal order for an ARIMA model.

    The ``auto_arima`` function seeks to identify the most optimal
    parameters for an ``ARIMA`` model, and returns a fitted ARIMA model. This
    function is based on the commonly-used R function,
    ``forecast::auto.arima`` [3].

    The ``auro_arima`` function works by conducting differencing tests (i.e.,
    Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller or
    Phillips–Perron) to determine the order of differencing, ``d``, and then
    fitting models within ranges of defined ``start_p``, ``max_p``,
    ``start_q``, ``max_q`` ranges. If the ``seasonal`` optional is enabled,
    ``auto_arima`` also seeks to identify the optimal ``P`` and ``Q`` hyper-
    parameters after conducting the Canova-Hansen to determine the optimal
    order of seasonal differencing, ``D``.

    In order to find the best model, ``auto_arima`` optimizes for a given
    ``information_criterion``, one of {'aic', 'aicc', 'bic', 'hqic', 'oob'}
    (Akaike Information Criterion, Corrected Akaike Information Criterion,
    Bayesian Information Criterion, Hannan-Quinn Information Criterion, or
    "out of bag"--for validation scoring--respectively) and returns the ARIMA
    which minimizes the value.

    Note that due to stationarity issues, ``auto_arima`` might not find a
    suitable model that will converge. If this is the case, a ``ValueError``
    will be thrown suggesting stationarity-inducing measures be taken prior
    to re-fitting or that a new range of ``order`` values be selected. Non-
    stepwise (i.e., essentially a grid search) selection can be slow,
    especially for seasonal data. Stepwise algorithm is outlined in Hyndman and
    Khandakar (2008).

    Parameters
    ----------
    y : array-like or iterable, shape=(n_samples,)
        The time-series to which to fit the ``ARIMA`` estimator. This may
        either be a Pandas ``Series`` object (statsmodels can internally
        use the dates in the index), or a numpy array. This should be a
        one-dimensional array of floats, and should not contain any
        ``np.nan`` or ``np.inf`` values.

    exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
        An optional 2-d array of exogenous variables. If provided, these
        variables are used as additional features in the regression
        operation. This should not include a constant or trend. Note that
        if an ``ARIMA`` is fit on exogenous features, it must be provided
        exogenous features for making predictions.

    start_p : int, optional (default=2)
        The starting value of ``p``, the order (or number of time lags)
        of the auto-regressive ("AR") model. Must be a positive integer.

    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value
        will automatically be selected based on the results of the ``test``
        (i.e., either the Kwiatkowski–Phillips–Schmidt–Shin, Augmented
        Dickey-Fuller or the Phillips–Perron test will be conducted to find
        the most probable value). Must be a positive integer or None. Note
        that if ``d`` is None, the runtime could be significantly longer.

    start_q : int, optional (default=2)
        The starting value of ``q``, the order of the moving-average
        ("MA") model. Must be a positive integer.

    max_p : int, optional (default=5)
        The maximum value of ``p``, inclusive. Must be a positive integer
        greater than or equal to ``start_p``.

    max_d : int, optional (default=2)
        The maximum value of ``d``, or the maximum number of non-seasonal
        differences. Must be a positive integer greater than or equal to ``d``.

    max_q : int, optional (default=5)
        The maximum value of ``q``, inclusive. Must be a positive integer
        greater than ``start_q``.

    start_P : int, optional (default=1)
        The starting value of ``P``, the order of the auto-regressive portion
        of the seasonal model.

    D : int, optional (default=None)
        The order of the seasonal differencing. If None (by default, the value
        will automatically be selected based on the results of the
        ``seasonal_test``. Must be a positive integer or None.

    start_Q : int, optional (default=1)
        The starting value of ``Q``, the order of the moving-average portion
        of the seasonal model.

    max_P : int, optional (default=2)
        The maximum value of ``P``, inclusive. Must be a positive integer
        greater than ``start_P``.

    max_D : int, optional (default=1)
        The maximum value of ``D``. Must be a positive integer greater
        than ``D``.

    max_Q : int, optional (default=2)
        The maximum value of ``Q``, inclusive. Must be a positive integer
        greater than ``start_Q``.

    max_order : int, optional (default=10)
        If the sum of ``p`` and ``q`` is >= ``max_order``, a model will
        *not* be fit with those parameters, but will progress to the next
        combination. Default is 5. If ``max_order`` is None, it means there
        are no constraints on maximum order.

    m : int, optional (default=1)
        The period for seasonal differencing, ``m`` refers to the number of
        periods in each season. For example, ``m`` is 4 for quarterly data, 12
        for monthly data, or 1 for annual (non-seasonal) data. Default is 1.
        Note that if ``m`` == 1 (i.e., is non-seasonal), ``seasonal`` will be
        set to False. For more information on setting this parameter, see
        :ref:`period`.

    seasonal : bool, optional (default=True)
        Whether to fit a seasonal ARIMA. Default is True. Note that if
        ``seasonal`` is True and ``m`` == 1, ``seasonal`` will be set to False.

    stationary : bool, optional (default=False)
        Whether the time-series is stationary and ``d`` should be set to zero.

    information_criterion : str, optional (default='aic')
        The information criterion used to select the best ARIMA model. One of
        ``pyramid.arima.auto_arima.VALID_CRITERIA``, ('aic', 'bic', 'hqic',
        'oob').

    alpha : float, optional (default=0.05)
        Level of the test for testing significance.

    test : str, optional (default='kpss')
        Type of unit root test to use in order to detect stationarity if
        ``stationary`` is False and ``d`` is None. Default is 'kpss'
        (Kwiatkowski–Phillips–Schmidt–Shin).

    seasonal_test : str, optional (default='ch')
        This determines which seasonal unit root test is used if ``seasonal``
        is True and ``D`` is None. Default is 'ch' (Canova-Hansen).

    stepwise : bool, optional (default=True)
        Whether to use the stepwise algorithm outlined in Hyndman and Khandakar
        (2008) to identify the optimal model parameters. The stepwise algorithm
        can be significantly faster than fitting all (or a ``random`` subset
        of) hyper-parameter combinations and is less likely to over-fit
        the model.

    n_jobs : int, optional (default=1)
        The number of models to fit in parallel in the case of a grid search
        (``stepwise=False``). Default is 1, but -1 can be used to designate
        "as many as possible".

    start_params : array-like, optional (default=None)
        Starting parameters for ``ARMA(p,q)``.  If None, the default is given
        by ``ARMA._fit_start_params``.

    transparams : bool, optional (default=True)
        Whether or not to transform the parameters to ensure stationarity.
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
        The args to pass to the constructor of the offset (``d``) test. See
        ``pyramid.arima.stationarity`` for more details.

    seasonal_test_args : dict, optional (default=None)
        The args to pass to the constructor of the seasonal offset (``D``)
        test. See ``pyramid.arima.seasonality`` for more details.

    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If
        ``suppress_warnings`` is True, all of the warnings coming from
        ``ARIMA`` will be squelched.

    error_action : str, optional (default='warn')
        If unable to fit an ``ARIMA`` due to stationarity issues, whether to
        warn ('warn'), raise the ``ValueError`` ('raise') or ignore ('ignore').
        Note that the default behavior is to warn, and fits that fail will be
        returned as None. This is the recommended behavior, as statsmodels
        ARIMA and SARIMAX models hit bugs periodically that can cause
        an otherwise healthy parameter combination to fail for reasons not
        related to pyramid.

    trace : bool, optional (default=False)
        Whether to print status on the fits. Note that this can be
        very verbose...

    random : bool, optional (default=False)
        Similar to grid searches, ``auto_arima`` provides the capability to
        perform a "random search" over a hyper-parameter space. If ``random``
        is True, rather than perform an exhaustive search or ``stepwise``
        search, only ``n_fits`` ARIMA models will be fit (``stepwise`` must be
        False for this option to do anything).

    random_state : int, long or numpy ``RandomState``, optional (default=None)
        The PRNG for when ``random=True``. Ensures replicable testing and
        results.

    n_fits : int, optional (default=10)
        If ``random`` is True and a "random search" is going to be performed,
        ``n_iter`` is the number of ARIMA models to be fit.

    return_valid_fits : bool, optional (default=False)
        If True, will return all valid ARIMA fits in a list. If False (by
        default), will only return the best fit.

    out_of_sample_size : int, optional (default=0)
        The ``ARIMA`` class can fit only a portion of the data if specified,
        in order to retain an "out of bag" sample score. This is the
        number of examples from the tail of the time series to hold out
        and use as validation examples. The model will not be fit on these
        samples, but the observations will be added into the model's ``endog``
        and ``exog`` arrays so that future forecast values originate from the
        end of the endogenous vector.

        For instance::

            y = [0, 1, 2, 3, 4, 5, 6]
            out_of_sample_size = 2

            > Fit on: [0, 1, 2, 3, 4]
            > Score on: [5, 6]
            > Append [5, 6] to end of self.arima_res_.data.endog values

    scoring : str, optional (default='mse')
        If performing validation (i.e., if ``out_of_sample_size`` > 0), the
        metric to use for scoring the out-of-sample data. One of {'mse', 'mae'}

    scoring_args : dict, optional (default=None)
        A dictionary of key-word arguments to be passed to the ``scoring``
        metric.

    **fit_args : dict, optional (default=None)
        A dictionary of keyword arguments to pass to the :func:`ARIMA.fit`
        method.

    See Also
    --------
    :func:`pyramid.arima.ARIMA`

    Notes
    -----
    Fitting with `stepwise=False` can prove slower, especially when
    `seasonal=True`.

    References
    ----------
    .. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average
    .. [2] R's auto-arima source code: http://bit.ly/2gOh5z2
    .. [3] R's auto-arima documentation: http://bit.ly/2wbBvUN
    """
    start = time.time()

    # validate start/max points
    if any(_ < 0 for _ in (max_p, max_q, max_P, max_Q, start_p,
                           start_q, start_P, start_Q)):
        raise ValueError('starting and max p, q, P & Q values must '
                         'be positive integers (>= 0)')
    if max_p < start_p or max_q < start_q \
            or max_P < start_P or max_Q < start_Q:
        raise ValueError('max p, q, P & Q must be >= than '
                         'their starting values')

    # validate max_order
    if max_order is None:
        max_order = np.inf
    elif max_order < 0:
        raise ValueError('max_order must be None or a positive integer (>= 0)')

    # alternatively, if the start_p and start_q supercede
    # the max order, that's a failable offense...
    if (not seasonal and start_p + start_q > max_order) or \
            (seasonal and start_P + start_p + start_Q + start_q > max_order):
        raise ValueError('If max_order is prescribed, it must '
                         'exceed sum of starting orders')

    # validate d & D
    for _d, _max_d in ((d, max_d), (D, max_D)):
        if _max_d < 0:
            raise ValueError('max_d & max_D must be positive integers (>= 0)')
        if _d is not None:
            if _d < 0:
                raise ValueError('d & D must be None or a positive '
                                 'integer (>= 0)')
            # v0.9.0+ - ignore this if it's explicitly set...
            # if _d > _max_d:
            #     raise ValueError('if explicitly defined, d & D must be <= '
            #                      'max_d & <= max_D, respectively')

    # is stepwise AND parallel enabled?
    if stepwise and n_jobs != 1:
        n_jobs = 1
        warnings.warn('stepwise model cannot be fit in parallel (n_jobs=%i). '
                      'Falling back to stepwise parameter search.' % n_jobs)

    # check on m
    if m < 1:
        raise ValueError('m must be a positive integer (> 0)')

    # check on n_iter
    if random and n_fits < 0:
        raise ValueError('n_iter must be a positive integer '
                         'for a random search')

    # validate error action
    actions = {'warn', 'raise', 'ignore', None}
    if error_action not in actions:
        raise ValueError('error_action must be one of %r, but got %r'
                         % (actions, error_action))

    # copy array
    y = column_or_1d(check_array(y, ensure_2d=False, dtype=DTYPE, copy=True,
                                 force_all_finite=True))  # type: np.ndarray
    n_samples = y.shape[0]

    # check for constant data
    if is_constant(y):
        warnings.warn('Input time-series is completely constant; '
                      'returning a (0, 0, 0) ARMA.')
        return _return_wrapper(_post_ppc_arima(
            _fit_arima(y, xreg=exogenous, order=(0, 0, 0), seasonal_order=None,
                       start_params=start_params, trend=trend, method=method,
                       transparams=transparams, solver=solver, maxiter=maxiter,
                       disp=disp, callback=callback, fit_params=fit_args,
                       suppress_warnings=suppress_warnings, trace=trace,
                       error_action=error_action, scoring=scoring,
                       out_of_sample_size=out_of_sample_size,
                       scoring_args=scoring_args)),
            return_valid_fits, start, trace)

    # test ic, and use AIC if n <= 3
    if information_criterion not in VALID_CRITERIA:
        raise ValueError('auto_arima not defined for information_criteria=%s. '
                         'Valid information criteria include: %r'
                         % (information_criterion, VALID_CRITERIA))

    # the R code handles this, but I don't think statsmodels
    # will even fit a model this small...
    # if n_samples <= 3:
    #     if information_criterion != 'aic':
    #         warnings.warn('n_samples (%i) <= 3 '
    #                       'necessitates using AIC' % n_samples)
    #     information_criterion = 'aic'

    # adjust max p, q -- R code:
    # max.p <- min(max.p, floor(serieslength/3))
    # max.q <- min(max.q, floor(serieslength/3))
    max_p = int(min(max_p, np.floor(n_samples / 3)))
    max_q = int(min(max_q, np.floor(n_samples / 3)))

    # this is not in the R code and poses a risk that R did not consider...
    # if max_p|q has now dropped below start_p|q, correct it.
    start_p = min(start_p, max_p)
    start_q = min(start_q, max_q)

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
        seasonal_test_args = seasonal_test_args if seasonal_test_args is \
                                                   not None else dict()
        D = nsdiffs(xx, m=m, test=seasonal_test, max_D=max_D,
                    **seasonal_test_args)

        if D > 0 and exogenous is not None:
            diffxreg = diff(exogenous, differences=D, lag=m)
            # check for constance on any column
            if np.apply_along_axis(is_constant, arr=diffxreg, axis=0).any():
                D -= 1

    # D might still be None if not seasonal. Py 3 will throw and error for that
    # unless we explicitly check for ``seasonal``
    if D > 0:
        dx = diff(xx, differences=D, lag=m)
    else:
        dx = xx

    # If D was too big, we might have gotten rid of x altogether!
    if dx.shape[0] == 0:
        raise ValueError("The seasonal differencing order, D=%i, was too "
                         "large for your time series, and after differencing, "
                         "there are no samples remaining in your data. "
                         "Try a smaller value for D, or if you didn't set D "
                         "to begin with, try setting it explicitly. This can "
                         "also occur in seasonal settings when m is too large."
                         % D)

    # difference the exogenous matrix
    if exogenous is not None:
        if D > 0:
            diffxreg = diff(exogenous, differences=D, lag=m)
        else:
            diffxreg = exogenous
    else:
        # here's the thing... we're only going to use diffxreg if exogenous
        # was not None in the first place. However, PyCharm doesn't know that
        # and it thinks we might use it before assigning it. Therefore, assign
        # it to None as a default value and it won't raise the warning anymore.
        diffxreg = None

    # determine/set the order of differencing by estimating the number of
    # orders it would take in order to make the TS stationary.
    if d is None:
        offset_test_args = offset_test_args if offset_test_args is \
                                               not None else dict()
        d = ndiffs(dx, test=test, alpha=alpha, max_d=max_d, **offset_test_args)

        if d > 0 and exogenous is not None:
            diffxreg = diff(diffxreg, differences=d, lag=1)

            # if any columns are constant, subtract one order of differencing
            if np.apply_along_axis(is_constant, arr=diffxreg, axis=0).any():
                d -= 1

    # check differences (do we want to warn?...)
    if error_action == 'warn' and not suppress_warnings:
        if D >= 2:
            warnings.warn("Having more than one seasonal differences is "
                          "not recommended. Please consider using only one "
                          "seasonal difference.")
        # if D is -1, this will be off, so we include the OR
        elif D + d > 2 or d > 2:
            warnings.warn("Having 3 or more differencing operations is not "
                          "recommended. Please consider reducing the total "
                          "number of differences.")

    if d > 0:
        dx = diff(dx, differences=d, lag=1)

    # check for constance
    if is_constant(dx):
        if exogenous is None and not (D > 0 or d < 2):
            raise ValueError('data follow a simple polynomial and are not '
                             'suitable for ARIMA modeling')

        # perfect regression
        ssn = None if not seasonal else (0, D, 0, m)
        return _return_wrapper(
            _post_ppc_arima(_fit_arima(y, xreg=exogenous, order=(0, d, 0),
                                       seasonal_order=ssn,
                                       start_params=start_params, trend=trend,
                                       method=method, transparams=transparams,
                                       solver=solver, maxiter=maxiter,
                                       disp=disp, callback=callback,
                                       fit_params=fit_args,
                                       suppress_warnings=suppress_warnings,
                                       trace=trace,
                                       error_action=error_action,
                                       scoring=scoring,
                                       out_of_sample_size=out_of_sample_size,
                                       scoring_args=scoring_args)),
            return_valid_fits, start, trace)

    # seasonality issues
    if m > 1:
        if max_P > 0:
            max_p = min(max_p, m - 1)
        if max_Q > 0:
            max_q = min(max_q, m - 1)

    # generate the set of (p, q, P, Q) FIRST, since it is contingent
    # on whether or not the user is interested in a seasonal ARIMA result.
    # This will reduce the search space for non-seasonal ARIMA models.
    # loop p, q. Make sure to loop at +1 interval,
    # since max_{p|q} is inclusive.
    if seasonal:
        gen = (
            ((p, d, q), (P, D, Q, m))
            for p in xrange(start_p, max_p + 1)
            for q in xrange(start_q, max_q + 1)
            for P in xrange(start_P, max_P + 1)
            for Q in xrange(start_Q, max_Q + 1)
            if p + q + P + Q <= max_order
        )
    else:
        # otherwise it's not seasonal, and we don't need the seasonal pieces
        gen = (
            ((p, d, q), None)
            for p in xrange(start_p, max_p + 1)
            for q in xrange(start_q, max_q + 1)
            if p + q <= max_order
        )

    if not stepwise:
        # if we are fitting a random search rather than an exhaustive one, we
        # will scramble up the generator (as a list) and only fit n_iter ARIMAs
        if random:
            random_state = check_random_state(random_state)

            # make a list to scramble...
            gen = random_state.permutation(list(gen))[:n_fits]

        # get results in parallel
        all_res = Parallel(n_jobs=n_jobs)(
            delayed(_fit_arima)(y, xreg=exogenous, order=order,
                                seasonal_order=seasonal_order,
                                start_params=start_params, trend=trend,
                                method=method, transparams=transparams,
                                solver=solver, maxiter=maxiter, disp=disp,
                                callback=callback,
                                fit_params=fit_args,
                                suppress_warnings=suppress_warnings,
                                trace=trace, error_action=error_action,
                                out_of_sample_size=out_of_sample_size,
                                scoring=scoring, scoring_args=scoring_args)
            for order, seasonal_order in gen)

    # otherwise, we're fitting the stepwise algorithm...
    else:
        if n_samples < 10:
            start_p = min(start_p, 1)
            start_q = min(start_q, 1)
            start_P = start_Q = 0

        # adjust to p, q, P, Q vals
        p = start_p = min(start_p, max_p)
        q = start_q = min(start_q, max_q)
        P = start_P = min(start_P, max_P)
        Q = start_Q = min(start_Q, max_Q)

        # init the stepwise model wrapper
        stepwise_wrapper = _StepwiseFitWrapper(
            y, xreg=exogenous, start_params=start_params, trend=trend,
            method=method, transparams=transparams, solver=solver,
            maxiter=maxiter, disp=disp, callback=callback, fit_params=fit_args,
            suppress_warnings=suppress_warnings, trace=trace,
            error_action=error_action, out_of_sample_size=out_of_sample_size,
            scoring=scoring, scoring_args=scoring_args, p=p, d=d, q=q,
            P=P, D=D, Q=Q, m=m, start_p=start_p, start_q=start_q,
            start_P=start_P, start_Q=start_Q, max_p=max_p, max_q=max_q,
            max_P=max_P, max_Q=max_Q, seasonal=seasonal,
            information_criterion=information_criterion,
            max_order=max_order)

        # fit a baseline p, d, q model and then a null model
        stepwise_wrapper.fit_increment_k_cache_set(True)  # p, d, q, P, D, Q
        stepwise_wrapper.fit_increment_k_cache_set(
            True, p=0, q=0, P=0, Q=0)  # null model

        # fit a basic AR model
        stepwise_wrapper.fit_increment_k_cache_set(
            max_p > 0 or max_P > 0, p=int(max_p > 0),
            q=0, P=int(m > 1 and max_P > 0), Q=0)

        # fit a basic MA model now
        stepwise_wrapper.fit_increment_k_cache_set(
            max_q > 0 or max_Q > 0, p=0, q=int(max_q > 0),
            P=0, Q=int(m > 1 and max_Q > 0))

        # might not be 4 if p, q etc. are already 0
        # assert stepwise_wrapper.k == 4  # sanity check...

        # do the step-through...
        all_res = stepwise_wrapper.step_through()

    # filter the non-successful ones
    filtered = _post_ppc_arima(all_res)

    # sort by the criteria - lower is better for both AIC and BIC
    # (https://stats.stackexchange.com/questions/81427/aic-guidelines-in-model-selection)
    sorted_res = sorted(filtered,
                        key=(lambda mod:
                             getattr(mod, information_criterion)()))

    # remove all the cached .pmdpkl files... someday write this as an exit hook
    # in case of a KeyboardInterrupt or anything
    for model in sorted_res:
        model._clear_cached_state()

    return _return_wrapper(sorted_res, return_valid_fits, start, trace)


class _StepwiseFitWrapper(object):
    """The stepwise algorithm fluctuates the more sensitive pieces of the ARIMA
    (the seasonal components) first, adjusting towards the direction of the
    smaller {A|B|HQ}IC(c), and continues to step down as long as the error
    shrinks. As long as the error term decreases and the best parameters have
    not shifted to a point where they can no longer change, ``k`` will
    increase, and the models will continue to be fit until the ``max_k`` is
    reached.

    References
    ----------
    .. [1] R's auto-arima stepwise source code: http://bit.ly/2vOma0W
    """
    def __init__(self, y, xreg, start_params, trend, method, transparams,
                 solver, maxiter, disp, callback, fit_params,
                 suppress_warnings, trace, error_action, out_of_sample_size,
                 scoring, scoring_args, p, d, q, P, D, Q, m, start_p, start_q,
                 start_P, start_Q, max_p, max_q, max_P, max_Q, seasonal,
                 information_criterion, max_order):
        # todo: I really hate how congested this block is, and just for the
        #       sake of a stateful __init__... Could we just pass **kwargs
        #       (MUCH less expressive...) in here? It would be much more
        #       difficult to debug later...

        # stuff for the fit call
        self.y = y
        self.xreg = xreg
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.transparams = transparams
        self.solver = solver
        self.maxiter = maxiter
        self.disp = disp
        self.callback = callback
        self.fit_params = fit_params
        self.suppress_warnings = suppress_warnings
        self.trace = trace
        self.error_action = error_action
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.information_criterion = information_criterion

        # order stuff
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.seasonal = seasonal
        self.max_order = max_order

        # other internal start vars
        self.bestfit = None
        self.k = self.start_k = 0
        self.max_k = 100

        # results list to store intermittent hashes of orders to determine if
        # we've seen this order before...
        self.results_dict = dict()  # type: dict[tuple, ARIMA]

        # define the info criterion getter ONCE to avoid multiple lambda
        # creation calls
        self.get_ic = (lambda mod: getattr(mod, self.information_criterion)())

    def is_new_better(self, new_model):
        if self.bestfit is None:
            return True
        elif new_model is None:
            return False

        current_ic, new_ic = self.get_ic(self.bestfit), self.get_ic(new_model)
        return new_ic < current_ic

    # this function takes a boolean expression, fits & caches a model,
    # increments k, and then sets the new value for p, d, P, Q, etc.
    def fit_increment_k_cache_set(self, expr, p=None, q=None, P=None, Q=None):
        # extract p, q, P, Q
        p = self.p if p is None else p
        q = self.q if q is None else q
        P = self.P if P is None else P
        Q = self.Q if Q is None else Q

        order = (p, self.d, q)
        ssnl = (P, self.D, Q, self.m) if self.seasonal else None

        # if the sum of the orders > max_order we do not build this model...
        order_sum = p + q + (P + Q if self.seasonal else 0)

        # if all conditions hold true, good to build this model.
        if expr and order_sum <= self.max_order and (order, ssnl) not \
                in self.results_dict:
            self.k += 1
            # cache the fit
            fit = _fit_arima(self.y, xreg=self.xreg, order=order,
                             seasonal_order=ssnl,
                             start_params=self.start_params, trend=self.trend,
                             method=self.method, transparams=self.transparams,
                             solver=self.solver, maxiter=self.maxiter,
                             disp=self.disp, callback=self.callback,
                             fit_params=self.fit_params,
                             suppress_warnings=self.suppress_warnings,
                             trace=self.trace, error_action=self.error_action,
                             out_of_sample_size=self.out_of_sample_size,
                             scoring=self.scoring,
                             scoring_args=self.scoring_args)

            # use the orders as a key to be hashed for
            # the dictionary (pointing to fit)
            self.results_dict[(order, ssnl)] = fit

            if self.is_new_better(fit):
                self.bestfit = fit

                # reset p, d, P, Q, etc.
                self.p, self.q, self.P, self.Q = p, q, P, Q

    def step_through(self):
        while self.start_k < self.k < self.max_k:
            self.start_k = self.k

            # Each of these fit the models for an expression, a new p,
            # q, P or Q, and then reset best models
            # This takes the place of a lot of copy/pasted code....

            # P fluctuations:
            self.fit_increment_k_cache_set(self.P > 0, P=self.P - 1)
            self.fit_increment_k_cache_set(self.P < self.max_P, P=self.P + 1)

            # Q fluctuations:
            self.fit_increment_k_cache_set(self.Q > 0, Q=self.Q - 1)
            self.fit_increment_k_cache_set(self.Q < self.max_Q, Q=self.Q + 1)

            # P & Q fluctuations:
            self.fit_increment_k_cache_set(self.Q > 0 and self.P > 0,
                                           P=self.P - 1, Q=self.Q - 1)
            self.fit_increment_k_cache_set(self.Q < self.max_Q and
                                           self.P < self.max_P, P=self.P + 1,
                                           Q=self.Q + 1)

            # p fluctuations:
            self.fit_increment_k_cache_set(self.p > 0, p=self.p - 1)
            self.fit_increment_k_cache_set(self.p < self.max_p, p=self.p + 1)

            # q fluctuations:
            self.fit_increment_k_cache_set(self.q > 0, q=self.q - 1)
            self.fit_increment_k_cache_set(self.q < self.max_q, q=self.q + 1)

            # p & q fluctuations:
            self.fit_increment_k_cache_set(self.p > 0 and self.q > 0,
                                           q=self.q - 1, p=self.p - 1)
            self.fit_increment_k_cache_set(self.q < self.max_q and
                                           self.p < self.max_p, q=self.q + 1,
                                           p=self.p + 1)

        # return the values
        return self.results_dict.values()


def _fit_arima(x, xreg, order, seasonal_order, start_params, trend,
               method, transparams, solver, maxiter, disp, callback,
               fit_params, suppress_warnings, trace, error_action,
               out_of_sample_size, scoring, scoring_args):
    start = time.time()
    try:
        fit = ARIMA(order=order, seasonal_order=seasonal_order,
                    start_params=start_params, trend=trend, method=method,
                    transparams=transparams, solver=solver, maxiter=maxiter,
                    disp=disp, callback=callback,
                    suppress_warnings=suppress_warnings,
                    out_of_sample_size=out_of_sample_size, scoring=scoring,
                    scoring_args=scoring_args)\
            .fit(x, exogenous=xreg, **fit_params)

    # for non-stationarity errors or singular matrices, return None
    except (LinAlgError, ValueError) as v:
        if error_action == 'warn':
            warnings.warn(_fmt_warning_str(order, seasonal_order),
                          ModelFitWarning)
        elif error_action == 'raise':
            # todo: can we do something more informative in case
            # the error is not on the pyramid side?
            raise v
        # if it's 'ignore' or 'warn', we just return None
        fit = None

    # do trace
    if trace:
        print('Fit ARIMA: %s; AIC=%.3f, BIC=%.3f, Fit time=%.3f seconds'
              % (_fmt_order_info(order, seasonal_order),
                 fit.aic() if fit is not None else np.nan,
                 fit.bic() if fit is not None else np.nan,
                 time.time() - start if fit is not None else np.nan))

    return fit


def _fmt_order_info(order, seasonal_order):
    return 'order=(%i, %i, %i)%s' \
           % (order[0], order[1], order[2],
              '' if seasonal_order is None
              else ' seasonal_order=(%i, %i, %i, %i)'
              % (seasonal_order[0], seasonal_order[1],
                 seasonal_order[2], seasonal_order[3]))


def _fmt_warning_str(order, seasonal_order):
    """This is just so we can test that the string will format
    even if we don't want the warnings in the tests
    """
    return ('Unable to fit ARIMA for %s; data is likely non-stationary. '
            '(if you do not want to see these warnings, run '
            'with error_action="ignore")'
            % _fmt_order_info(order, seasonal_order))


def _post_ppc_arima(a):
    """If there are no suitable models, raise a ValueError.
    Otherwise, return ``a``. In the case that ``a`` is an iterable
    (i.e., it made it to the end of the function), this method will
    filter out the None values and assess whether the list is empty.

    Parameters
    ----------
    a : ARIMA or iterable
        The list or ARIMAs, or an ARIMA
    """
    # if it's a result of making it to the end, it will
    # be a list of ARIMA models. Filter out the Nones
    # (the failed models)...
    if is_iterable(a):
        a = [m for m in a if m is not None]

    # if the list is empty, or if it was an ARIMA and it's None
    if not a:  # check for truthiness rather than None explicitly
        raise ValueError('Could not successfully fit ARIMA to input data. '
                         'It is likely your data is non-stationary. Please '
                         'induce stationarity or try a different '
                         'range of model order params. If your data is '
                         'seasonal, check the period (m) of the data.')
    # good to return
    return a


def _return_wrapper(fits, return_all, start, trace):
    """If the user wants to get all of the models back, this will
    return a list of the ARIMA models, otherwise it will just return
    the model. If this is called from the end of the function, ``fits``
    will already be a list.

    We *know* that if a function call makes it here, ``fits`` is NOT None
    or it would have thrown an exception in :func:`_post_ppc_arima`.

    Parameters
    ----------
    fits : iterable or ARIMA
        The ARIMA(s)

    return_all : bool
        Whether to return all.
    """
    # make sure it's an iterable
    if not is_iterable(fits):
        fits = [fits]

    # whether to print the final runtime
    if trace:
        print('Total fit time: %.3f seconds' % (time.time() - start))

    # which to return? if not all, then first index (assume sorted)
    if not return_all:
        return fits[0]
    return fits
