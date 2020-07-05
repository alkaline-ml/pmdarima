# -*- coding: utf-8 -*-

from pmdarima.arima import auto
from pmdarima.compat.pytest import pytest_error_str
import pytest


def test_deprecation_warnings():
    kwargs = {'transparams': True, 'method': 'lbfgs'}
    with pytest.warns(DeprecationWarning) as we:
        kwargs = auto._warn_for_deprecations(**kwargs)
    assert kwargs['method']
    assert 'transparams' not in kwargs
    assert we


def test_deprecation_warnings_on_class():
    with pytest.warns(DeprecationWarning) as we:
        auto.AutoARIMA(sarimax_kwargs={"simple_differencing": True})
    assert we


def test_issue_341():
    y = [0, 132, 163, 238, 29, 0, 150, 320, 249, 224, 197, 31, 0, 154,
         143, 132, 135, 158, 21, 0, 126, 100, 137, 105, 104, 8, 0, 165,
         191, 234, 253, 155, 25, 0, 228, 234, 265, 205, 191, 19, 0, 188,
         156, 172, 173, 166, 28, 0, 209, 160, 159, 129, 124, 18, 0, 155]

    with pytest.raises(ValueError) as ve:
        auto.auto_arima(
            y,
            start_p=1,
            start_q=1,
            test='adf',
            max_p=3,
            max_q=3,
            m=52,
            start_P=0,
            seasonal=True,
            d=None,
            D=1,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

    # assert that we catch the np LinAlg error and reraise with a more
    # meaningful message
    assert "Encountered exception in stationarity test" in pytest_error_str(ve)
