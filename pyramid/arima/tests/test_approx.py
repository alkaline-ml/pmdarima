# Test the approximation function

from __future__ import absolute_import
from pyramid.arima.approx import approx, _regularize
from pyramid.utils.array import c
from numpy.testing import assert_array_almost_equal
import numpy as np

table = c(0.216, 0.176, 0.146, 0.119)
tablep = c(0.01, 0.025, 0.05, 0.10)
stat = 1.01


def test_regularize():
    x, y = c(0.5, 0.5, 1.0, 1.5), c(1, 2, 3, 4)
    x, y = _regularize(x, y, 'mean')

    assert_array_almost_equal(x, np.array([0.5, 1.0, 1.5]))
    assert_array_almost_equal(y, np.array([1.5, 3.0, 4.0]))


def test_approx_rule1():
    # for rule = 1
    x, y = approx(table, tablep, stat, rule=1)
    assert_array_almost_equal(x, c(1.01))
    assert_array_almost_equal(y, c(np.nan))


def test_approx_rule2():
    # for rule = 2
    x, y = approx(table, tablep, stat, rule=2)
    assert_array_almost_equal(x, c(1.01))
    assert_array_almost_equal(y, c(0.01))
