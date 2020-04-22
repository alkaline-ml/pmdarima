# -*- coding: utf-8 -*-

from pmdarima.arima import ARIMA
from pmdarima.compat.pytest import pytest_error_str
from pmdarima.compat import sklearn as sk

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
import pytest


@pytest.mark.parametrize(
    'x,i,exp', [
        pytest.param(np.array([1, 2, 3, 4, 5]), [0, 1], np.array([1, 2])),
        pytest.param(pd.Series([1, 2, 3, 4, 5]), [0, 1], np.array([1, 2])),
        pytest.param(np.array([[1, 2], [3, 4]]), [0], np.array([[1, 2]])),
    ]
)
def test_safe_indexing(x, i, exp):
    res = sk.safe_indexing(x, i)
    if hasattr(res, "values"):  # pd.Series
        res = res.values
    assert_array_equal(exp, res)


def test_check_is_fitted_error():
    with pytest.raises(TypeError) as te:
        sk.check_is_fitted(None, None)
    assert "attributes must be a string or iterable" in pytest_error_str(te)


def test_not_fitted_error():
    with pytest.raises(sk.NotFittedError) as nfe:
        mod = ARIMA((0, 1, 0))
        sk.check_is_fitted(mod, "arima_res_")
    assert "Model has not been fit!" in pytest_error_str(nfe)
