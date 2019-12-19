# -*- coding: utf-8 -*-

from numpy.testing import assert_array_almost_equal

from pmdarima.preprocessing import LogEndogTransformer
from pmdarima.preprocessing import BoxCoxEndogTransformer


def test_same():
    y = [1, 2, 3]
    trans = BoxCoxEndogTransformer(lmbda=0)
    log_trans = LogEndogTransformer()
    y_t, _ = trans.fit_transform(y)
    log_y_t, _ = log_trans.fit_transform(y)
    assert_array_almost_equal(log_y_t, y_t)


def test_invertible():
    y = [1, 2, 3]
    trans = LogEndogTransformer()
    y_t, _ = trans.fit_transform(y)
    y_prime, _ = trans.inverse_transform(y_t)
    assert_array_almost_equal(y, y_prime)
