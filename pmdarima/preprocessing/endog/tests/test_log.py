# -*- coding: utf-8 -*-

from numpy.testing import assert_array_almost_equal
from sklearn.base import clone

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


def test_log_clone_issue_407():
    # https://github.com/alkaline-ml/pmdarima/issues/407
    log = LogEndogTransformer(lmbda=10)
    res, _ = log.fit_transform([0, 10])

    # we swap lmbda2 and lmbda internally
    assert log.lmbda2 == 10
    assert log.lmbda == 0

    log2 = clone(log)
    assert log2.lmbda2 == 10
    assert log2.lmbda == 0
    res2, _ = log2.fit_transform([0, 10])

    assert_array_almost_equal(res, res2)
