# -*- coding: utf-8 -*-
#
# Methods for optimizing auto-arima

from numpy.linalg import LinAlgError
import numpy as np

import time
import warnings

from .arima import ARIMA
from .warnings import ModelFitWarning


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
                 information_criterion, max_order, with_intercept):
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
        self.with_intercept = with_intercept

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
        self.results_dict = dict()  # dict[tuple -> ARIMA]

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
                             scoring_args=self.scoring_args,
                             with_intercept=self.with_intercept)

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
               out_of_sample_size, scoring, scoring_args, with_intercept):
    start = time.time()
    try:
        fit = ARIMA(order=order, seasonal_order=seasonal_order,
                    start_params=start_params, trend=trend, method=method,
                    transparams=transparams, solver=solver, maxiter=maxiter,
                    disp=disp, callback=callback,
                    suppress_warnings=suppress_warnings,
                    out_of_sample_size=out_of_sample_size, scoring=scoring,
                    scoring_args=scoring_args,
                    with_intercept=with_intercept)\
            .fit(x, exogenous=xreg, **fit_params)

    # for non-stationarity errors or singular matrices, return None
    except (LinAlgError, ValueError) as v:
        if error_action == 'warn':
            warnings.warn(_fmt_warning_str(order, seasonal_order),
                          ModelFitWarning)
        elif error_action == 'raise':
            # todo: can we do something more informative in case
            # the error is not on the pmdarima side?
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
