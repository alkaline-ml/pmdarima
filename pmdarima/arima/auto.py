# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Automatically find optimal parameters for an ARIMA

from __future__ import absolute_import

from joblib import Parallel, delayed
import numpy as np
from sklearn.utils.validation import check_array, column_or_1d, check_is_fitted
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
import time
import warnings

from ..base import BaseARIMA
from . import _doc
from .utils import ndiffs, is_constant, nsdiffs
from ..utils import diff, is_iterable
from ..utils.metaestimators import if_has_delegate
from ._auto_solvers import _fit_arima, _StepwiseFitWrapper

# for python 3 compat
from ..compat.python import xrange
from ..compat.numpy import DTYPE

__all__ = [
    'auto_arima',
    'AutoARIMA'
]

# The valid information criteria
VALID_CRITERIA = {'aic', 'aicc', 'bic', 'hqic', 'oob'}


class AutoARIMA(BaseARIMA):
    # Don't add the y, exog, etc. here since they are used in 'fit'
    __doc__ = _doc._AUTO_ARIMA_DOCSTR.format(
        y="", exogenous="", fit_args="", return_valid_fits="")

    # todo: someday store defaults somewhere else for single source of truth
    def __init__(self, start_p=2, d=None, start_q=2, max_p=5,
                 max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
                 max_D=1, max_Q=2, max_order=10, m=1, seasonal=True,
                 stationary=False, information_criterion='aic', alpha=0.05,
                 test='kpss', seasonal_test='ocsb', stepwise=True, n_jobs=1,
                 start_params=None, trend=None, method=None, transparams=True,
                 solver='lbfgs', maxiter=None, disp=0, callback=None,
                 offset_test_args=None, seasonal_test_args=None,
                 suppress_warnings=False, error_action='warn', trace=False,
                 random=False, random_state=None, n_fits=10,
                 out_of_sample_size=0, scoring='mse',
                 scoring_args=None, with_intercept=True):

        self.start_p = start_p
        self.d = d
        self.start_q = start_q
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.start_P = start_P
        self.D = D
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.max_order = max_order
        self.m = m
        self.seasonal = seasonal
        self.stationary = stationary
        self.information_criterion = information_criterion
        self.alpha = alpha
        self.test = test
        self.seasonal_test = seasonal_test
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.transparams = transparams
        self.solver = solver
        self.maxiter = maxiter
        self.disp = disp
        self.callback = callback
        self.offset_test_args = offset_test_args
        self.seasonal_test_args = seasonal_test_args
        self.suppress_warnings = suppress_warnings
        self.error_action = error_action
        self.trace = trace
        self.random = random
        self.random_state = random_state
        self.n_fits = n_fits
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.with_intercept = with_intercept

    def fit(self, y, exogenous=None, **fit_args):
        """Fit the auto-arima estimator

        Fit an AutoARIMA to a vector, ``y``, of observations with an
        optional matrix of ``exogenous`` variables.

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

        **fit_args : dict or kwargs
            Any keyword arguments to pass to the auto-arima function.
        """
        self.model_ = auto_arima(
            y, exogenous=exogenous, start_p=self.start_p, d=self.d,
            start_q=self.start_q, max_p=self.max_p, max_d=self.max_d,
            max_q=self.max_q, start_P=self.start_P, D=self.D,
            start_Q=self.start_Q, max_P=self.max_P, max_D=self.max_D,
            max_Q=self.max_Q, max_order=self.max_order, m=self.m,
            seasonal=self.seasonal, stationary=self.stationary,
            information_criterion=self.information_criterion,
            alpha=self.alpha, test=self.test, seasonal_test=self.seasonal_test,
            stepwise=self.stepwise, n_jobs=self.n_jobs,
            start_params=self.start_params, trend=self.trend,
            method=self.method, transparams=self.transparams,
            solver=self.solver, maxiter=self.maxiter, disp=self.disp,
            callback=self.callback, offset_test_args=self.offset_test_args,
            seasonal_test_args=self.seasonal_test_args,
            suppress_warnings=self.suppress_warnings,
            error_action=self.error_action, trace=self.trace,
            random=self.random, random_state=self.random_state,
            n_fits=self.n_fits,
            return_valid_fits=False,  # only return ONE
            out_of_sample_size=self.out_of_sample_size, scoring=self.scoring,
            scoring_args=self.scoring_args, with_intercept=self.with_intercept,
            **fit_args)

        return self

    @if_has_delegate("model_")
    def predict_in_sample(self, exogenous=None, start=None,
                          end=None, dynamic=False):
        return self.model_.predict_in_sample(
            exogenous=exogenous, start=start, end=end, dynamic=dynamic)

    @if_has_delegate("model_")
    def predict(self, n_periods=10, exogenous=None,
                return_conf_int=False, alpha=0.05):
        return self.model_.predict(
            n_periods=n_periods, exogenous=exogenous,
            return_conf_int=return_conf_int, alpha=alpha)

    @if_has_delegate("model_")
    def update(self, y, exogenous=None, maxiter=None, **kwargs):
        return self.model_.update(
            y, exogenous=exogenous, maxiter=maxiter, **kwargs)

    # TODO: decorator to automate all this composition + AIC, etc.


def auto_arima(y, exogenous=None, start_p=2, d=None, start_q=2, max_p=5,
               max_d=2, max_q=5, start_P=1, D=None, start_Q=1, max_P=2,
               max_D=1, max_Q=2, max_order=10, m=1, seasonal=True,
               stationary=False, information_criterion='aic', alpha=0.05,
               test='kpss', seasonal_test='ocsb', stepwise=True, n_jobs=1,
               start_params=None, trend=None, method=None, transparams=True,
               solver='lbfgs', maxiter=None, disp=0, callback=None,
               offset_test_args=None, seasonal_test_args=None,
               suppress_warnings=False, error_action='warn', trace=False,
               random=False, random_state=None, n_fits=10,
               return_valid_fits=False, out_of_sample_size=0, scoring='mse',
               scoring_args=None, with_intercept=True, **fit_args):
    # NOTE: Doc is assigned BELOW this function

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
                       scoring_args=scoring_args,
                       with_intercept=with_intercept)),
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
        seasonal_test_args = seasonal_test_args \
            if seasonal_test_args is not None else dict()
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
        offset_test_args = offset_test_args \
            if offset_test_args is not None else dict()
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
                                       scoring_args=scoring_args,
                                       with_intercept=with_intercept)),
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
                                scoring=scoring, scoring_args=scoring_args,
                                with_intercept=with_intercept)
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
            max_order=max_order, with_intercept=with_intercept)

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


# Assign the doc to the auto_arima func
auto_arima.__doc__ = _doc._AUTO_ARIMA_DOCSTR.format(
    y=_doc._Y_DOCSTR,
    exogenous=_doc._EXOG_DOCSTR,
    fit_args=_doc._FIT_ARGS_DOCSTR,
    return_valid_fits=_doc._VALID_FITS_DOCSTR
)


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
