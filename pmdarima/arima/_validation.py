# -*- coding: utf-8 -*-

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
