# -*- coding: utf-8 -*-

from pmdarima.arima import ARIMA
from pmdarima.model_selection._split import RollingForecastCV, \
    SlidingWindowForecastCV
from pmdarima.model_selection._validation import cross_val_score, \
    cross_validate
from pmdarima.datasets import load_wineind
import pytest
import numpy as np

y = load_wineind()


@pytest.mark.parametrize('cv', [
    SlidingWindowForecastCV(window_size=100, step=24, h=1),
    RollingForecastCV(initial=150, step=12, h=1),
])
@pytest.mark.parametrize(
    'est', [
        ARIMA(order=(2, 1, 1), seasonal_order=(0, 0, 0, 1)),
        ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
    ]
)
@pytest.mark.parametrize('verbose', [0, 2, 4])
def test_cv_scores(cv, est, verbose):
    scores = cross_val_score(
        est, y, scoring='mean_squared_error', cv=cv, verbose=verbose)
    assert isinstance(scores, np.ndarray)
