# -*- coding: utf-8 -*-

from pmdarima.arima import auto
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
