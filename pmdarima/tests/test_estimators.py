# -*- coding: utf-8 -*-

from sklearn.base import clone
from pmdarima.arima import ARIMA, AutoARIMA
from pmdarima.pipeline import Pipeline
from pmdarima.datasets import load_wineind
from pmdarima.preprocessing import FourierFeaturizer
import pytest

y = load_wineind()


@pytest.mark.parametrize(
    'est', [
        ARIMA(order=(2, 1, 1)),
        AutoARIMA(seasonal=False, maxiter=3),
        Pipeline([
            ("fourier", FourierFeaturizer(m=12)),
            ("arima", AutoARIMA(seasonal=False, stepwise=True,
                                suppress_warnings=True, d=1, max_p=2, max_q=0,
                                start_q=0, start_p=1,
                                maxiter=3, error_action='ignore'))
        ])
    ]
)
def test_clonable(est):
    # fit it, then clone it
    est.fit(y)
    est2 = clone(est)
    assert isinstance(est2, est.__class__)
    assert est is not est2
