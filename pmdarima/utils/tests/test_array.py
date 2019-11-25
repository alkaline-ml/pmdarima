
from __future__ import absolute_import

from pmdarima.utils.array import diff, diff_inv, c, is_iterable, as_series, check_exog
from pmdarima.utils import get_callable

from numpy.testing import assert_array_equal, assert_array_almost_equal

import pytest
import pandas as pd
import numpy as np

x = np.arange(5)
m = np.array([10, 5, 12, 23, 18, 3, 2, 0, 12]).reshape(3, 3).T
X = pd.DataFrame.from_records(
    np.random.RandomState(2).rand(4, 4),
    columns=['a', 'b', 'c', 'd']
)

# need some infinite values in X for testing check_exog
X_nan = X.copy()
X_nan.loc[0, 'a'] = np.nan

X_inf = X.copy()
X_inf.loc[0, 'a'] = np.inf


def test_diff():
    # test vector for lag = (1, 2), diff = (1, 2)
    assert_array_equal(diff(x, lag=1, differences=1), np.ones(4))
    assert_array_equal(diff(x, lag=1, differences=2), np.zeros(3))
    assert_array_equal(diff(x, lag=2, differences=1), np.ones(3) * 2)
    assert_array_equal(diff(x, lag=2, differences=2), np.zeros(1))

    # test matrix for lag = (1, 2), diff = (1, 2)
    assert_array_equal(diff(m, lag=1, differences=1),
                       np.array([[-5, -5, -2], [7, -15, 12]]))
    assert_array_equal(diff(m, lag=1, differences=2),
                       np.array([[12, -10, 14]]))
    assert_array_equal(diff(m, lag=2, differences=1), np.array([[2, -20, 10]]))
    assert diff(m, lag=2, differences=2).shape[0] == 0


def test_diff_inv():
    # test vector for lag = (1, 2), diff = (1, 2)
    # The below values are the output of the following R:
    # > x <- c(0, 1, 2, 3, 4)
    # > diffinv(x, lag=1, differences=1)
    # [1]  0  0  1  3  6 10
    # > diffinv(x, lag=1, differences=2)
    # [1]  0  0  0  1  4 10 20
    # > diffinv(x, lag=2, differences=1)
    # [1] 0 0 0 1 2 4 6
    # > diffinv(x, lag=2, differences=2)
    # [1] 0 0 0 0 0 1 2 5 8
    assert_array_equal(diff_inv(x, lag=1, differences=1),
                       np.array([0., 0., 1., 3., 6., 10.]))
    assert_array_equal(diff_inv(x, lag=1, differences=2),
                       np.array([0., 0., 0., 1., 4., 10., 20.]))
    assert_array_equal(diff_inv(x, lag=2, differences=1),
                       np.array([0., 0., 0., 1., 2., 4., 6.]))
    assert_array_equal(diff_inv(x, lag=2, differences=2),
                       np.array([0., 0., 0., 0., 0., 1., 2., 5., 8.]))


def test_concatenate():
    assert_array_equal(c(1, np.zeros(3)), np.array([1.0, 0.0, 0.0, 0.0]))
    assert_array_equal(c([1], np.zeros(3)), np.array([1.0, 0.0, 0.0, 0.0]))
    assert_array_equal(c(1), np.ones(1))
    assert c() is None
    assert_array_equal(c([1]), np.ones(1))


def test_corner_in_callable():
    # test the ValueError in the get-callable method
    with pytest.raises(ValueError):
        get_callable('fake-key', {'a': 1})


def test_corner():
    # fails because lag < 1
    with pytest.raises(ValueError):
        diff(x=x, lag=0)


def test_is_iterable():
    assert not is_iterable("this string")
    assert is_iterable(["this", "list"])
    assert not is_iterable(None)
    assert is_iterable(np.array([1, 2]))


def test_as_series():
    assert isinstance(as_series([1, 2, 3]), pd.Series)
    assert isinstance(as_series(np.arange(5)), pd.Series)
    assert isinstance(as_series(pd.Series([1, 2, 3])), pd.Series)


@pytest.mark.parametrize(
    'arr', [
        np.random.rand(5),
        pd.Series(np.random.rand(5)),
    ]
)
def test_check_exog_ndim_value_err(arr):
    with pytest.raises(ValueError):
        check_exog(arr)


@pytest.mark.parametrize('arr', [X_nan, X_inf])
def test_check_exog_infinite_value_err(arr):
    with pytest.raises(ValueError):
        check_exog(arr, force_all_finite=True)

    # show it passes when False
    assert check_exog(
        arr, force_all_finite=False, dtype=None, copy=False) is arr


def test_exog_pd_dataframes():
    # test with copy
    assert check_exog(X, force_all_finite=True, copy=True).equals(X)

    # test without copy
    assert check_exog(X, force_all_finite=True, copy=False) is X


def test_exog_np_array():
    X_np = np.random.RandomState(1).rand(5, 5)

    # show works on a list
    assert_array_almost_equal(X_np, check_exog(X_np.tolist()))
    assert_array_almost_equal(X_np, check_exog(X_np))
