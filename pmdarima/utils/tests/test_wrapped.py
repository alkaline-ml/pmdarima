# -*- coding: utf-8 -*-

import pmdarima as pm
from pmdarima.utils.wrapped import acf, pacf

import statsmodels.api as sm
import numpy as np

import pytest

y = pm.datasets.load_wineind()


@pytest.mark.parametrize(
    'wrapped_func,native_func', [
        pytest.param(sm.tsa.stattools.acf, acf),
        pytest.param(sm.tsa.stattools.pacf, pacf)
    ])
def test_wrapped_functions(wrapped_func, native_func):
    sm_res = wrapped_func(y)  # type: np.ndarray
    pm_res = native_func(y)
    assert np.allclose(sm_res, pm_res)

    # Show the docstrings are the same
    assert wrapped_func.__doc__ == native_func.__doc__
