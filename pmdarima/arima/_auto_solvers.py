# -*- coding: utf-8 -*-
#
# Methods for optimizing auto-arima

from numpy.linalg import LinAlgError
import numpy as np

import time
import warnings
import traceback

from .arima import ARIMA
from .warnings import ModelFitWarning
from ._context import ContextType, ContextStore
from . import _validation
from ..compat import statsmodels as sm_compat
from datetime import datetime
import functools


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


class _StepwiseFitWrapper:
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
    .. [2] https://robjhyndman.com/hyndsight/arma-roots/
    """
    def __init__(self, y, xreg, start_params, trend, method, maxiter,
                 fit_params, suppress_warnings, trace, error_action,
                 out_of_sample_size, scoring, scoring_args,
                 p, d, q, P, D, Q, m, start_p, start_q,
                 start_P, start_Q, max_p, max_q, max_P, max_Q, seasonal,
                 information_criterion, with_intercept, **kwargs):

        self.trace = _validation.check_trace(trace)

        # Create a partial of the fit call so we don't have arg bloat all over
        self._fit_arima = functools.partial(
            _fit_candidate_model,
            x=y,
            xreg=xreg,
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
            do_root_test=True,
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
        self.start_p = start_p
        self.start_q = start_q
        self.start_P = start_P
        self.start_Q = start_Q
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

            fit, _, new_ic = self._fit_arima(
                order=order,
                seasonal_order=seasonal_order,
                with_intercept=constant)

            # use the orders as a key to be hashed for
            # the dictionary (pointing to fit)
            self.results_dict[(order, seasonal_order, constant)] = fit

            # cache this so we can lookup best model IC downstream
            self.ic_dict[(order, seasonal_order, constant)] = new_ic

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

    def solve_stepwise(self):
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
                # TODO: wonder if it's worth a trace message here?
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

            # The old code was much more concise and stateful, but was also
            # much more difficult to read and debug. This was rewritten for
            # v1.5.0 in an effort to make it more simple to decipher.
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

        if self.trace > 1 and self.bestfit:
            print("Final model order: (%i, %i, %i)x(%i, %i, %i, %i) "
                  "(constant=%s)"
                  % (self.bestfit.order[0],
                     self.bestfit.order[1],
                     self.bestfit.order[2],
                     self.bestfit.seasonal_order[0],
                     self.bestfit.seasonal_order[1],
                     self.bestfit.seasonal_order[2],
                     self.bestfit.seasonal_order[3],
                     self.bestfit.with_intercept))

            # TODO: if (allowdrift || allowmean)

        # check if the search has been ended after max_steps
        if self.exec_context.max_steps is not None \
                and self.k >= self.exec_context.max_steps:
            warnings.warn('stepwise search has reached the maximum number '
                          'of tries to find the best fit model')

        # TODO: if (approximation && !is.null(bestfit$arma))

        if not self.bestfit:
            warnings.warn("No viable models found")

        return _sort_and_filter_fits(self.results_dict, self.ic_dict)


def _sort_and_filter_fits(results_dict, ic_dict):
    # return the sorted values from the dicts
    filtered_models_ics = sorted(
        [(v, ic_dict[k])
         for k, v in results_dict.items()
         if v is not None],
        key=(lambda fit_ic: fit_ic[1]),
    )

    # TODO: can we break ties in sorting by using the more simple model?

    if not filtered_models_ics:
        raise ValueError(
            "Could not successfully fit a viable ARIMA model "
            "to input data using the stepwise algorithm.\nSee "
            "http://alkaline-ml.com/pmdarima/no-successful-model.html "
            "for more information on why this can happen."
        )

    # (fit_a, fit_b), (np.inf, 0.5), etc.
    fits, _ = zip(*filtered_models_ics)
    return fits


def _fit_candidate_model(x,
                         xreg,
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
                         do_root_test,
                         **kwargs):
    """Instantiate and fit a candidate model

    1. Initialize a model
    2. Fit model
    3. Perform a root test
    4. Return model, information criterion
    """
    debug_str = _arima_debug_str(order, seasonal_order, with_intercept)

    start = time.time()
    fit_time = np.nan
    ic = np.inf
    fit = None

    try:
        fit = ARIMA(order=order, seasonal_order=seasonal_order,
                    start_params=start_params, trend=trend, method=method,
                    maxiter=maxiter, suppress_warnings=suppress_warnings,
                    out_of_sample_size=out_of_sample_size, scoring=scoring,
                    scoring_args=scoring_args,
                    with_intercept=with_intercept, **kwargs)\
            .fit(x, exogenous=xreg, **fit_params)

    # for non-stationarity errors or singular matrices, return None
    except (LinAlgError, ValueError) as v:
        if error_action == "raise":
            raise v

        elif error_action in ("warn", "trace"):
            warning_str = 'Error fitting %s ' \
                          '(if you do not want to see these warnings, run ' \
                          'with error_action="ignore").' \
                          % debug_str

            if error_action == 'trace':
                warning_str += "\nTraceback:\n" + traceback.format_exc()

            warnings.warn(warning_str, ModelFitWarning)

    else:
        fit_time = time.time() - start
        ic = getattr(fit, information_criterion)()  # aic, bic, aicc, etc.

        # check the roots of the new model, and set IC to inf if the
        # roots are near non-invertible
        if do_root_test:
            ic = _root_test(fit, ic, trace)

    # log the model fit
    if trace:
        print(
            "{model}   : {ic_name}={ic:.3f}, Time={time:.2f} sec"
            .format(model=debug_str,
                    ic_name=information_criterion.upper(),
                    ic=ic,
                    time=fit_time)
        )

    return fit, fit_time, ic


def _arima_debug_str(order, seasonal_order, with_intercept):
    p, d, q = order
    P, D, Q, m = seasonal_order
    int_str = "intercept"
    return (
        " ARIMA({p},{d},{q})({P},{D},{Q})[{m}] {intercept}".format(
            p=p,
            d=d,
            q=q,
            P=P,
            D=D,
            Q=Q,
            m=m,
            # just for consistent spacing
            intercept=int_str if with_intercept else " " * len(int_str)
        )
    )
