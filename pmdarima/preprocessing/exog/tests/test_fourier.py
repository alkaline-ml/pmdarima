# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal

from pmdarima.preprocessing.exog import FourierFeaturizer
from pmdarima.compat.pytest import pytest_error_str
import pmdarima as pm

import pytest

wineind = pm.datasets.load_wineind()


class TestFourierREquivalency:

    # The following R code is what we want to reproduce:
    #   > set.seed(99)
    #   > n = 20
    #   > m = 5
    #   > y <- ts(rnorm(n) + (1:n)%%100/30, f=m)
    #   > library(forecast)
    #   > exog = fourier(y, K=2)
    #   > head(exog, 2)
    #             S1-5      C1-5       S2-5      C2-5
    #   [1,] 0.9510565  0.309017  0.5877853 -0.809017
    #   [2,] 0.5877853 -0.809017 -0.9510565  0.309017

    y = pm.c(
        0.24729584, 0.54632480, 0.18782870, 0.57719184, -0.19617125,
        0.32267403, -0.63051185, 0.75629093, -0.06411691, -0.96090867,
        -0.37910238, 1.32155036, 1.18338768, -2.04188735, -2.54093410,
        0.53359913, 0.17264767, -1.14502766, 1.13196478, 0.93762046)

    expected = np.array([
        [0.9510565, 0.309017, 0.5877853, -0.809017],
        [0.5877853, -0.809017, -0.9510565, 0.309017],
        [-0.5877853, -0.809017, 0.9510565, 0.309017],
        [-0.9510565, 0.309017, -0.5877853, -0.809017],
        [0.0000000, 1.000000, 0.0000000, 1.000000],
        [0.9510565, 0.309017, 0.5877853, -0.809017],
        [0.5877853, -0.809017, -0.9510565, 0.309017],
        [-0.5877853, -0.809017, 0.9510565, 0.309017],
        [-0.9510565, 0.309017, -0.5877853, -0.809017],
        [0.0000000, 1.000000, 0.0000000, 1.000000],
        [0.9510565, 0.309017, 0.5877853, -0.809017],
        [0.5877853, -0.809017, -0.9510565, 0.309017],
        [-0.5877853, -0.809017, 0.9510565, 0.309017],
        [-0.9510565, 0.309017, -0.5877853, -0.809017],
        [0.0000000, 1.000000, 0.0000000, 1.000000],
        [0.9510565, 0.309017, 0.5877853, -0.809017],
        [0.5877853, -0.809017, -0.9510565, 0.309017],
        [-0.5877853, -0.809017, 0.9510565, 0.309017],
        [-0.9510565, 0.309017, -0.5877853, -0.809017],
        [0.0000000, 1.000000, 0.0000000, 1.000000],
    ])

    @pytest.mark.parametrize(
        'X', [
            None,
            np.random.rand(y.shape[0], 3)
        ]
    )
    def test_r_equivalency(self, X):
        y = self.y
        expected = self.expected

        trans = FourierFeaturizer(m=5, k=2).fit(y)
        _, xreg = trans.transform(y, X)

        # maybe subset
        if hasattr(xreg, 'iloc'):
            xreg = xreg.values
        assert_array_almost_equal(expected, xreg[:, -4:])

        # maybe assert on X
        if X is not None:
            assert_array_almost_equal(X, xreg[:, :3])

            # Test a bad forecast (X dim does not match n_periods dim)
            with pytest.raises(ValueError):
                trans.transform(y, X=np.random.rand(5, 3), n_periods=2)


def test_hyndman_blog():
    # This is the exact code Hyndman ran in his blog post on the matter:
    # https://robjhyndman.com/hyndsight/longseasonality/
    n = 2000
    m = 200
    y = np.random.RandomState(1).normal(size=n) + \
        (np.arange(1, n + 1) % 100 / 30)

    trans = FourierFeaturizer(m=m, k=5).fit(y)
    _, xreg = trans.transform(y)

    arima = pm.auto_arima(y,
                          X=xreg,
                          seasonal=False,
                          maxiter=2,  # very short
                          start_p=4,
                          max_p=5,
                          d=0,
                          max_q=1,
                          start_q=0,
                          simple_differencing=True)  # type: pm.ARIMA

    # Show we can forecast 10 in the future
    _, xreg_test = trans.transform(y, n_periods=10)
    arima.predict(n_periods=10, X=xreg_test)


def test_update_transform():
    n = 150
    m = 10
    y = np.random.RandomState(1).normal(size=n) + \
        (np.arange(1, n + 1) % 100 / 30)

    train, test = y[:100], y[100:]

    trans = FourierFeaturizer(m=m, k=5).fit(train)
    _, xreg = trans.transform(train)

    # Now update with the test set and show the xreg is diff
    yt, Xt = trans.update_and_transform(test, X=None)
    assert yt is test
    assert Xt.shape[0] == test.shape[0]
    assert trans.n_ == y.shape[0]

    # Now assert that if we do a vanilla transform with no n_periods, the last
    # 50 are the same as the Xt we just got and the first 100 are the same as
    # we got earlier
    _, xreg2 = trans.transform(y)
    assert_array_almost_equal(xreg2[:100], xreg)
    assert_array_almost_equal(xreg2[100:], Xt)


def test_value_error_check():
    feat = FourierFeaturizer(m=12)
    with pytest.raises(ValueError) as ve:
        feat._check_y_X(wineind, None, null_allowed=False)
    assert 'non-None' in pytest_error_str(ve)


def test_value_error_on_fit():
    feat = FourierFeaturizer(m=12, k=8)
    with pytest.raises(ValueError) as ve:
        feat.fit_transform(wineind)
    assert 'k must be' in pytest_error_str(ve)
