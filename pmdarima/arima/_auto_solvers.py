# -*- coding: utf-8 -*-
#
# Methods for optimizing auto-arima

from numpy.linalg import LinAlgError
import numpy as np

from datetime import datetime
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

import abc
import functools
import time
import warnings
import traceback

from .arima import ARIMA
from ..warnings import ModelFitWarning
from ._context import ContextType, ContextStore
from . import _validation
from ..compat import statsmodels as sm_compat


def _root_test(model, ic, trace):
    """
    Check the roots of the new model, and set IC to inf if the roots are
    near non-invertible. This is a little bit different than how Rob does it:
    https://github.com/robjhyndman/forecast/blob/master/R/newarima2.R#L780
    In our test, we look directly at the inverse roots to see if they come
    anywhere near the unit circle border
    """
    max_invroot = 0
    p, d, q = model.order
    P, D, Q, m = model.seasonal_order

    if p + P > 0:
        max_invroot = max(0, *np.abs(1 / model.arroots()))
    if q + Q > 0 and np.isfinite(ic):
        max_invroot = max(0, *np.abs(1 / model.maroots()))

    if max_invroot > 1 - 1e-2:
        ic = np.inf
        if trace > 1:
            print(
                "Near non-invertible roots for order "
                "(%i, %i, %i)(%i, %i, %i, %i); setting score to inf (at "
                "least one inverse root too close to the border of the "
                "unit circle: %.3f)"
                % (p, d, q, P, D, Q, m, max_invroot))
    return ic


class _SolverMixin(metaclass=abc.ABCMeta):
    """The solver interface implemented by wrapper classes"""

    @abc.abstractmethod
    def solve(self):
        """Must be implemented by subclasses"""


class _RandomFitWrapper(_SolverMixin):
    """Searches for the best model using a random search"""

    def __init__(self, y, X, fit_partial, d, D, m, max_order,
                 max_p, max_q, max_P, max_Q, random, random_state,
                 n_fits, n_jobs, seasonal, trace, with_intercept,
                 sarimax_kwargs):

        # NOTE: pre-1.5.2, we started at start_p, start_q, etc. However, when
        # using stepwise=FALSE in R, hyndman starts at 0. He only uses start_*
        # when stepwise=TRUE.

        # generate the set of (p, q, P, Q) FIRST, since it is contingent
        # on whether or not the user is interested in a seasonal ARIMA result.
        # This will reduce the search space for non-seasonal ARIMA models.
        # loop p, q. Make sure to loop at +1 interval,
        # since max_{p|q} is inclusive.
        if seasonal:
            gen = (
                ((p, d, q), (P, D, Q, m))
                for p in range(0, max_p + 1)
                for q in range(0, max_q + 1)
                for P in range(0, max_P + 1)
                for Q in range(0, max_Q + 1)
                if p + q + P + Q <= max_order
            )
        else:
            # otherwise it's not seasonal and we don't need the seasonal pieces
            gen = (
                ((p, d, q), (0, 0, 0, 0))
                for p in range(0, max_p + 1)
                for q in range(0, max_q + 1)
                if p + q <= max_order
            )

        # if we are fitting a random search rather than an exhaustive one, we
        # will scramble up the generator (as a list) and only fit n_fits ARIMAs
        if random:
            random_state = check_random_state(random_state)

            # make a list to scramble... `gen` may have a ragged nested
            # sequence, so we have to explicitly use dtype='object', otherwise
            # it will raise a ValueError on numpy >= 1.24
            gen = random_state.permutation(np.array(list(gen), dtype='object'))[:n_fits]  # noqa: E501

        self.gen = gen
        self.n_jobs = n_jobs
        self.trace = trace

        # New partial containing y, X
        self.fit_partial = functools.partial(
            fit_partial,
            y=y,
            X=X,
            with_intercept=with_intercept,
            **sarimax_kwargs,
        )

    def solve(self):
        """Do a random search"""
        fit_partial = self.fit_partial
        n_jobs = self.n_jobs
        gen = self.gen

        # get results in parallel
        all_res = Parallel(n_jobs=n_jobs)(
            delayed(fit_partial)(
                order=order,
                seasonal_order=seasonal_order,
            )
            for order, seasonal_order in gen
        )

        sorted_fits = _sort_and_filter_fits(all_res)
        if self.trace and sorted_fits:
            print(f"\nBest model: {str(sorted_fits[0])}")

        return sorted_fits


class _StepwiseFitWrapper(_SolverMixin):
    """Searches for the best model using the stepwise algorithm.
    
    The stepwise algorithm fluctuates the more sensitive pieces of the ARIMA
    (the seasonal components) first, adjusting towards the direction of the
    smaller {A|B|HQ}IC(c), and continues to step down as long as the error
    shrinks. As long as the error term decreases and the best parameters have
    not shifted to a point where they can no longer change, ``k`` will
    increase, and the models will continue to be fit until the ``max_k`` is
    reached.

    References
    ----------
    .. [1] R's auto-arima stepwise source code: https://github.com/robjhyndman/forecast/blob/30308a4e314ff29338291462e81bf68ff0c5f86d/R/newarima2.R#L366
    .. [2] https://robjhyndman.com/hyndsight/arma-roots/
    """  # noqa
    def __init__(self, y, X, start_params, trend, method, maxiter,
                 fit_params, suppress_warnings, trace, error_action,
                 out_of_sample_size, scoring, scoring_args,
                 p, d, q, P, D, Q, m, max_p, max_q, max_P, max_Q, seasonal,
                 information_criterion, with_intercept, **kwargs):

        self.trace = _validation.check_trace(trace)

        # Create a partial of the fit call so we don't have arg bloat all over
        self._fit_arima = functools.partial(
            _fit_candidate_model,
            y=y,
            X=X,
            start_params=start_params,
            trend=trend,
            method=method,
            maxiter=maxiter,
            fit_params=fit_params,
            suppress_warnings=suppress_warnings,
            trace=self.trace,
            error_action=error_action,
            out_of_sample_size=out_of_sample_size,
            scoring=scoring,
            scoring_args=scoring_args,
            information_criterion=information_criterion,
            **kwargs)

        self.information_criterion = information_criterion
        self.with_intercept = with_intercept

        # order stuff we will be incrementing
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.seasonal = seasonal

        # execution context passed through the context manager
        self.exec_context = ContextStore.get_or_empty(ContextType.STEPWISE)

        # other internal start vars
        self.k = self.start_k = 0
        self.max_k = 100 if self.exec_context.max_steps is None \
            else self.exec_context.max_steps
        self.max_dur = self.exec_context.max_dur

        # results list to store intermittent hashes of orders to determine if
        # we've seen this order before...
        self.results_dict = dict()  # dict[tuple -> ARIMA]
        self.ic_dict = dict()  # dict[tuple -> float]
        self.fit_time_dict = dict()  # dict[tuple -> float]

        self.bestfit = None
        self.bestfit_key = None  # (order, seasonal_order, constant)

    def _do_fit(self, order, seasonal_order, constant=None):
        """Do a fit and determine whether the model is better"""
        if not self.seasonal:
            seasonal_order = (0, 0, 0, 0)
        seasonal_order = sm_compat.check_seasonal_order(seasonal_order)

        # we might be fitting without a constant
        if constant is None:
            constant = self.with_intercept

        if (order, seasonal_order, constant) not in self.results_dict:

            # increment the number of fits
            self.k += 1

            fit, fit_time, new_ic = self._fit_arima(
                order=order,
                seasonal_order=seasonal_order,
                with_intercept=constant)

            # use the orders as a key to be hashed for
            # the dictionary (pointing to fit)
            self.results_dict[(order, seasonal_order, constant)] = fit

            # cache this so we can lookup best model IC downstream
            self.ic_dict[(order, seasonal_order, constant)] = new_ic
            self.fit_time_dict[(order, seasonal_order, constant)] = fit_time

            # Determine if the new fit is better than the existing fit
            if fit is None or np.isinf(new_ic):
                return False

            # no benchmark model
            if self.bestfit is None:
                self.bestfit = fit
                self.bestfit_key = (order, seasonal_order, constant)

                if self.trace > 1:
                    print("First viable model found (%.3f)" % new_ic)
                return True

            # otherwise there's a current best
            current_ic = self.ic_dict[self.bestfit_key]
            if new_ic < current_ic:

                if self.trace > 1:
                    print("New best model found (%.3f < %.3f)"
                          % (new_ic, current_ic))

                self.bestfit = fit
                self.bestfit_key = (order, seasonal_order, constant)
                return True

        # we've seen this model before
        return False

    def solve(self):
        start_time = datetime.now()
        p, d, q = self.p, self.d, self.q
        P, D, Q, m = self.P, self.D, self.Q, self.m
        max_p, max_q = self.max_p, self.max_q
        max_P, max_Q = self.max_P, self.max_Q

        if self.trace:
            print("Performing stepwise search to minimize %s"
                  % self.information_criterion)

        # fit a baseline p, d, q model
        self._do_fit((p, d, q), (P, D, Q, m))

        # null model with possible constant
        if self._do_fit((0, d, 0), (0, D, 0, m)):
            p = q = P = Q = 0

        # A basic AR model
        if max_p > 0 or max_P > 0:
            _p = 1 if max_p > 0 else 0
            _P = 1 if (m > 1 and max_P > 0) else 0
            if self._do_fit((_p, d, 0), (_P, D, 0, m)):
                p = _p
                P = _P
                q = Q = 0

        # Basic MA model
        if max_q > 0 or max_Q > 0:
            _q = 1 if max_q > 0 else 0
            _Q = 1 if (m > 1 and max_Q > 0) else 0
            if self._do_fit((0, d, _q), (0, D, _Q, m)):
                p = P = 0
                Q = _Q
                q = _q

        # Null model with NO constant (if we haven't tried it yet)
        if self.with_intercept:
            if self._do_fit((0, d, 0), (0, D, 0, m), constant=False):
                p = q = P = Q = 0

        while self.start_k < self.k < self.max_k:
            self.start_k = self.k

            # break loop if execution time exceeds the timeout threshold. needs
            # to be at front of loop, since a single pass may reach termination
            # criteria by end and we only want to warn and break if the loop is
            # continuing again
            dur = (datetime.now() - start_time).total_seconds()
            if self.max_dur and dur > self.max_dur:
                warnings.warn('early termination of stepwise search due to '
                              'max_dur threshold (%.3f > %.3f)'
                              % (dur, self.max_dur))
                break

            # NOTE: k changes for every fit, so we might need to bail halfway
            # through the loop, hence the multiple checks.
            if P > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P - 1, D, Q, m)):
                P -= 1
                continue

            if Q > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P, D, Q - 1, m)):
                Q -= 1
                continue

            if P < max_P and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P + 1, D, Q, m)):
                P += 1
                continue

            if Q < max_Q and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P, D, Q + 1, m)):
                Q += 1
                continue

            if Q > 0 and P > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P - 1, D, Q - 1, m)):
                Q -= 1
                P -= 1
                continue

            if Q < max_Q and P > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P - 1, D, Q + 1, m)):
                Q += 1
                P -= 1
                continue

            if Q > 0 and P < max_P and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P + 1, D, Q - 1, m)):
                Q -= 1
                P += 1
                continue

            if Q < max_Q and P < max_P and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q), (P + 1, D, Q + 1, m)):
                Q += 1
                P += 1
                continue

            if p > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p - 1, d, q), (P, D, Q, m)):
                p -= 1
                continue

            if q > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q - 1), (P, D, Q, m)):
                q -= 1
                continue

            if p < max_p and \
                    self.k < self.max_k and \
                    self._do_fit((p + 1, d, q), (P, D, Q, m)):
                p += 1
                continue

            if q < max_q and \
                    self.k < self.max_k and \
                    self._do_fit((p, d, q + 1), (P, D, Q, m)):
                q += 1
                continue

            if q > 0 and p > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p - 1, d, q - 1), (P, D, Q, m)):
                q -= 1
                p -= 1
                continue

            if q < max_q and p > 0 and \
                    self.k < self.max_k and \
                    self._do_fit((p - 1, d, q + 1), (P, D, Q, m)):
                q += 1
                p -= 1
                continue

            if q > 0 and p < max_p and \
                    self.k < self.max_k and \
                    self._do_fit((p + 1, d, q - 1), (P, D, Q, m)):
                q -= 1
                p += 1
                continue

            if q < max_q and p < max_p and \
                    self.k < self.max_k and \
                    self._do_fit((p + 1, d, q + 1), (P, D, Q, m)):
                q += 1
                p += 1
                continue

            # R: if (allowdrift || allowmean)
            # we don't have these args, so we just default this case to true to
            # evaluate all corners
            if self.k < self.max_k and \
                    self._do_fit((p, d, q),
                                 (P, D, Q, m),
                                 constant=not self.with_intercept):
                self.with_intercept = not self.with_intercept
                continue

        # check if the search has been ended after max_steps
        if self.exec_context.max_steps is not None \
                and self.k >= self.exec_context.max_steps:
            warnings.warn('stepwise search has reached the maximum number '
                          'of tries to find the best fit model')

        # TODO: if (approximation && !is.null(bestfit$arma)) - refit best w MLE

        filtered_models_ics = sorted(
            [(v, self.fit_time_dict[k], self.ic_dict[k])
             for k, v in self.results_dict.items()
             if v is not None],
            key=(lambda fit_ic: fit_ic[1]),
        )

        sorted_fits = _sort_and_filter_fits(filtered_models_ics)
        if self.trace and sorted_fits:
            print(f"\nBest model: {str(sorted_fits[0])}")

        return sorted_fits


def _fit_candidate_model(y,
                         X,
                         order,
                         seasonal_order,
                         start_params,
                         trend,
                         method,
                         maxiter,
                         fit_params,
                         suppress_warnings,
                         trace,
                         error_action,
                         out_of_sample_size,
                         scoring,
                         scoring_args,
                         with_intercept,
                         information_criterion,
                         **kwargs):
    """Instantiate and fit a candidate model

    1. Initialize a model
    2. Fit model
    3. Perform a root test
    4. Return model, information criterion
    """
    start = time.time()
    fit_time = np.nan
    ic = np.inf

    # Fit outside try block, so if there is a type error in user input we
    # don't mask it with a warning or worse
    fit = ARIMA(order=order, seasonal_order=seasonal_order,
                start_params=start_params, trend=trend, method=method,
                maxiter=maxiter, suppress_warnings=suppress_warnings,
                out_of_sample_size=out_of_sample_size, scoring=scoring,
                scoring_args=scoring_args,
                with_intercept=with_intercept, **kwargs)

    try:
        fit.fit(y, X=X, **fit_params)

    # for non-stationarity errors or singular matrices, return None
    except (LinAlgError, ValueError) as v:
        if error_action == "raise":
            raise v

        elif error_action in ("warn", "trace"):
            warning_str = 'Error fitting %s ' \
                          '(if you do not want to see these warnings, run ' \
                          'with error_action="ignore").' \
                          % str(fit)

            if error_action == 'trace':
                warning_str += "\nTraceback:\n" + traceback.format_exc()

            warnings.warn(warning_str, ModelFitWarning)

    else:
        fit_time = time.time() - start
        ic = getattr(fit, information_criterion)()  # aic, bic, aicc, etc.

        # check the roots of the new model, and set IC to inf if the
        # roots are near non-invertible
        ic = _root_test(fit, ic, trace)

    # log the model fit
    if trace:
        print(
            "{model}   : {ic_name}={ic:.3f}, Time={time:.2f} sec"
            .format(model=str(fit),
                    ic_name=information_criterion.upper(),
                    ic=ic,
                    time=fit_time)
        )

    return fit, fit_time, ic


def _sort_and_filter_fits(models):
    """Sort the results in ascending order, by information criterion

    If there are no suitable models, raise a ValueError.
    Otherwise, return ``a``. In the case that ``a`` is an iterable
    (i.e., it made it to the end of the function), this method will
    filter out the None values and assess whether the list is empty.

    Parameters
    ----------
    models : tuple or list
        The list or (model, fit_time, information_criterion), or a single tuple
    """
    # if it's a result of making it to the end, it will be a list of models
    if not isinstance(models, list):
        models = [models]

    # Filter out the Nones or Infs (the failed models)...
    filtered = [(mod, ic) for mod, _, ic in models
                if mod is not None and np.isfinite(ic)]

    # if the list is empty, or if it was an ARIMA and it's None
    if not filtered:
        raise ValueError(
            "Could not successfully fit a viable ARIMA model "
            "to input data.\nSee "
            "http://alkaline-ml.com/pmdarima/no-successful-model.html "
            "for more information on why this can happen."
        )

    # sort by the criteria - lower is better for both AIC and BIC
    # (https://stats.stackexchange.com/questions/81427/aic-guidelines-in-model-selection)  # noqa
    sorted_res = sorted(filtered, key=(lambda mod_ic: mod_ic[1]))

    # TODO: break ties with fit time?
    models, _ = zip(*sorted_res)

    return models
