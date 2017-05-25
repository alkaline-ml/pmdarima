# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Tests for stationarity

from __future__ import print_function, absolute_import, division
import numpy as np

__all__ = [
    'adf_test',
    'is_constant',
    'kpss_test',
    'pp_test'
]


def is_constant(x):
    length = x.shape[0]
    return (x == (np.ones(length) * x[0])).all()


def kpss_test(x):
    """The Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test is used for testing
    a null hypothesis that an observable time series is stationary around a deterministic trend
    (i.e. trend-stationary) against the alternative of a unit root.
    """


def adf_test(x):
    """The Augmented Dickey-Fuller test
    """


def pp_test(x):
    """
    """


VALID_TESTS = dict(kpss=kpss_test,
                   adf=adf_test,
                   pp=pp_test)
