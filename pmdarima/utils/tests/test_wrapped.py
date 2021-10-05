# -*- coding: utf-8 -*-

import pmdarima as pm
from pmdarima.utils.wrapped import acf, pacf

import statsmodels.api as sm
import numpy as np

import pytest
from unittest import mock

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


@pytest.mark.parametrize("use_sm13", [True, False])
def test_statsmodels13_compatibility(use_sm13):
    with mock.patch("pmdarima.compat.statsmodels._use_sm13") as mock_sm:
        mock_sm.return_value = use_sm13

        wrapped_acf_result = acf(y)
        wrapped_pacf_result = pacf(y)

        if use_sm13:
            native_acf_result = sm.tsa.stattools.acf(y)
            native_pacf_result = sm.tsa.stattools.pacf(y)
        else:
            native_acf_result = sm.tsa.stattools.acf(y, nlags=40, fft=False)
            native_pacf_result = sm.tsa.stattools.pacf(y, nlags=40)

        assert np.allclose(wrapped_acf_result, native_acf_result)
        assert np.allclose(wrapped_pacf_result, native_pacf_result)
