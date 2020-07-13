# Test the approximation function

from pmdarima.arima.approx import approx, _regularize
from pmdarima.utils.array import c
from pmdarima.arima.stationarity import ADFTest

from numpy.testing import assert_array_almost_equal
import numpy as np

import pytest

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


@pytest.mark.parametrize(
    'kwargs', [

        # fails for length differences
        dict(x=[1, 2, 3], y=[1, 2], xout=1.0),

        # fails for bad string
        dict(x=table, y=table, xout=1.0, method='bad-string'),

        # fails for bad length
        dict(x=[], y=[], xout=[], ties='mean'),

        # fails for bad length
        dict(x=[], y=[], xout=[], method='constant'),

        # fails for linear when < 2 samples
        dict(x=[1], y=[1], xout=[], method='linear', ties='ordered'),

        # fails for bad length
        dict(x=[], y=[], xout=[], method='constant'),

    ]
)
def test_corner_errors(kwargs):
    with pytest.raises(ValueError):
        approx(**kwargs)


def test_valid_corner():
    # *doesn't* fail for constant when < 2 samples
    approx(x=[1], y=[1], xout=[], method='constant', ties='ordered')


def test_approx_precision():
    # Test an example from R vs. Python to compare the expected values and
    # make sure we get as close as possible. This is from an ADFTest where k=1
    # and x=austres
    tableipl = np.array([[-4.0664],
                         [-3.7468],
                         [-3.462],
                         [-3.1572],
                         [-1.2128],
                         [-0.8928],
                         [-0.6104],
                         [-0.2704]])

    _, interpol = approx(tableipl, ADFTest.tablep, xout=-1.337233, rule=2)
    assert np.allclose(interpol, 0.84880354)  # in R we get 0.8488036
