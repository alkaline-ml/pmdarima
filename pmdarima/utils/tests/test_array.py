
from pmdarima.utils.array import diff, diff_inv, c, is_iterable, as_series, \
    check_exog
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

# for diffinv
x_mat = (np.arange(9) + 1).reshape(3, 3).T


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


@pytest.mark.parametrize(
    'arr,lag,differences,xi,expected', [
        # VECTORS -------------------------------------------------------------
        # > x = c(0, 1, 2, 3, 4)
        # > diffinv(x, lag=1, differences=1)
        # [1]  0  0  1  3  6 10
        pytest.param(x, 1, 1, None, [0, 0, 1, 3, 6, 10]),

        # > diffinv(x, lag=1, differences=2)
        # [1]  0  0  0  1  4 10 20
        pytest.param(x, 1, 2, None, [0, 0, 0, 1, 4, 10, 20]),

        # > diffinv(x, lag=2, differences=1)
        # [1] 0 0 0 1 2 4 6
        pytest.param(x, 2, 1, None, [0, 0, 0, 1, 2, 4, 6]),

        # > diffinv(x, lag=2, differences=2)
        # [1] 0 0 0 0 0 1 2 5 8
        pytest.param(x, 2, 2, None, [0, 0, 0, 0, 0, 1, 2, 5, 8]),

        # This is a test of the intermediate stage when x == [1, 0, 3, 2]
        pytest.param([1, 0, 3, 2], 1, 1, [0], [0, 1, 1, 4, 6]),

        # This is an intermediate stage when x == [0, 1, 2, 3, 4]
        pytest.param(x, 1, 1, [0], [0, 0, 1, 3, 6, 10]),

        # MATRICES ------------------------------------------------------------
        # > matrix(data=c(1, 2, 3, 4, 5, 6, 7, 8, 9), nrow=3, ncol=3)
        #      [,1] [,2] [,3]
        # [1,]    1    4    7
        # [2,]    2    5    8
        # [3,]    3    6    9
        # > diffinv(X, 1, 1)
        #      [,1] [,2] [,3]
        # [1,]    0    0    0
        # [2,]    1    4    7
        # [3,]    3    9   15
        # [4,]    6   15   24
        pytest.param(x_mat, 1, 1, None,
                     [[0, 0, 0],
                      [1, 4, 7],
                      [3, 9, 15],
                      [6, 15, 24]]),

        # > diffinv(X, 1, 2)
        #      [,1] [,2] [,3]
        # [1,]    0    0    0
        # [2,]    0    0    0
        # [3,]    1    4    7
        # [4,]    4   13   22
        # [5,]   10   28   46
        pytest.param(x_mat, 1, 2, None,
                     [[0, 0, 0],
                      [0, 0, 0],
                      [1, 4, 7],
                      [4, 13, 22],
                      [10, 28, 46]]),

        # > diffinv(X, 2, 1)
        #      [,1] [,2] [,3]
        # [1,]    0    0    0
        # [2,]    0    0    0
        # [3,]    1    4    7
        # [4,]    2    5    8
        # [5,]    4   10   16
        pytest.param(x_mat, 2, 1, None,
                     [[0, 0, 0],
                      [0, 0, 0],
                      [1, 4, 7],
                      [2, 5, 8],
                      [4, 10, 16]]),

        # > diffinv(X, 2, 2)
        #      [,1] [,2] [,3]
        # [1,]    0    0    0
        # [2,]    0    0    0
        # [3,]    0    0    0
        # [4,]    0    0    0
        # [5,]    1    4    7
        # [6,]    2    5    8
        # [7,]    5   14   23
        pytest.param(x_mat, 2, 2, None,
                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1, 4, 7],
                      [2, 5, 8],
                      [5, 14, 23]]),
    ]
)
def test_diff_inv(arr, lag, differences, xi, expected):
    res = diff_inv(arr, lag=lag, differences=differences, xi=xi)
    expected = np.array(expected, dtype=np.float)
    assert_array_equal(expected, res)


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
    with pytest.raises(ValueError):
        diff_inv(x=x, lag=0)

    # fails because differences < 1
    with pytest.raises(ValueError):
        diff(x=x, differences=0)
    with pytest.raises(ValueError):
        diff_inv(x=x, differences=0)

    # Passing in xi with the incorrect shape to a 2-d array
    with pytest.raises(IndexError):
        diff_inv(x=np.array([[1, 1], [1, 1]]), xi=np.array([[1]]))


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
