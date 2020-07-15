# -*- coding: utf-8 -*-

"""
Arg validation for auto-arima calls. This allows us to test validation more
directly without having to fit numerous combinations of models.
"""

import numpy as np
import warnings

# The valid information criteria
VALID_CRITERIA = {'aic', 'aicc', 'bic', 'hqic', 'oob'}


def auto_intercept(with_intercept, default):
    """A more concise way to handle the default behavior of with_intercept"""
    if with_intercept == "auto":
        return default
    return with_intercept


def check_information_criterion(information_criterion, out_of_sample_size):
    """Check whether the information criterion is valid"""
    if information_criterion not in VALID_CRITERIA:
        raise ValueError('auto_arima not defined for information_criteria=%s. '
                         'Valid information criteria include: %r'
                         % (information_criterion, VALID_CRITERIA))

    # check on information criterion and out_of_sample size
    if information_criterion == 'oob' and out_of_sample_size == 0:
        information_criterion = 'aic'
        warnings.warn('information_criterion cannot be \'oob\' with '
                      'out_of_sample_size = 0. '
                      'Falling back to information criterion = aic.')

    return information_criterion


def check_kwargs(kwargs):
    """Return kwargs or an empty dict.

    We often pass named kwargs (like `sarimax_kwargs`) as None by default. This
    is to avoid a mutable default, which can bite you in unexpected ways. This
    will return a kwarg-compatible value.
    """
    if kwargs:
        return kwargs
    return {}


def check_m(m, seasonal):
    """Check the value of M (seasonal periodicity)"""
    if (m < 1 and seasonal) or m < 0:
        raise ValueError('m must be a positive integer (> 0)')

    if not seasonal:
        # default m is 1, so if it's the default, don't warn
        if m > 1:
            warnings.warn("m (%i) set for non-seasonal fit. Setting to 0" % m)
        m = 0

    return m


def check_n_jobs(stepwise, n_jobs):
    """Potentially update the n_jobs parameter

    We can't run in parallel with the stepwise algorithm. This checks
    ``n_jobs`` w.r.t. stepwise and will warn.
    """
    if stepwise and n_jobs != 1:
        n_jobs = 1
        warnings.warn('stepwise model cannot be fit in parallel (n_jobs=%i). '
                      'Falling back to stepwise parameter search.' % n_jobs)
    return n_jobs


def check_start_max_values(st, mx, argname):
    """Ensure starting points and ending points are valid"""
    if mx is None:
        mx = np.inf
    if st is None:
        raise ValueError("start_%s cannot be None" % argname)
    if st < 0:
        raise ValueError("start_%s must be positive" % argname)
    if mx < st:
        raise ValueError("max_%s must be >= start_%s" % (argname, argname))
    return st, mx


def check_trace(trace):
    """Check the value of trace"""
    if trace is None:
        return 0
    if isinstance(trace, (int, bool)):
        return int(trace)
    # otherwise just be truthy with it
    if trace:
        return 1
    return 0
