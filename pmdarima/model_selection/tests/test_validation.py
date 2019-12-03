# -*- coding: utf-8 -*-

from pmdarima.arima import ARIMA
from pmdarima.arima.warnings import ModelFitWarning
from pmdarima.compat.pytest import pytest_error_str
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima.model_selection._split import RollingForecastCV, \
    SlidingWindowForecastCV
from pmdarima.model_selection._validation import cross_val_score, \
    _check_scoring, cross_validate
from pmdarima.datasets import load_wineind
import pytest
import numpy as np
from unittest import mock

y = load_wineind()
exogenous = np.random.RandomState(1).rand(y.shape[0], 2)


@pytest.mark.parametrize('cv', [
    SlidingWindowForecastCV(window_size=100, step=24, h=1),
    RollingForecastCV(initial=150, step=12, h=1),
])
@pytest.mark.parametrize(
    'est', [
        ARIMA(order=(2, 1, 1), seasonal_order=(0, 0, 0, 1)),
        ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12)),
        Pipeline([
            ("fourier", FourierFeaturizer(m=12)),
            ("arima", ARIMA(order=(2, 1, 0), maxiter=3))
        ])
    ]
)
@pytest.mark.parametrize('verbose', [0, 2, 4])
@pytest.mark.parametrize('exog', [None, exogenous])
def test_cv_scores(cv, est, verbose, exog):
    scores = cross_val_score(
        est, y, exogenous=exog, scoring='mean_squared_error',
        cv=cv, verbose=verbose)
    assert isinstance(scores, np.ndarray)


def test_check_scoring():
    # This will work since it's a callable
    scorer = (lambda true, pred: np.nan)
    assert _check_scoring(scorer) is scorer

    # fails for bad metric
    with pytest.raises(ValueError):
        _check_scoring('bad metric')

    # fails for anything else
    with pytest.raises(TypeError):
        _check_scoring(123)


def test_model_error_returns_nan():
    with mock.patch('sklearn.base.clone', lambda x: x):
        mock_model = mock.MagicMock()

        def mock_fit(*args, **kwargs):
            raise ValueError()

        mock_model.fit = mock_fit

        with pytest.warns(ModelFitWarning):
            scores = cross_val_score(
                mock_model, y, scoring='mean_squared_error',
                cv=SlidingWindowForecastCV(window_size=100, step=24, h=1),
                verbose=0)

        assert np.isnan(scores).all()

        # if the error_score is 'raise', we will raise
        with pytest.raises(ValueError):
            cross_val_score(
                mock_model, y, scoring='mean_squared_error',
                cv=SlidingWindowForecastCV(window_size=100, step=24, h=1),
                verbose=0, error_score='raise')


def test_error_action_validation():
    est = ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
    with pytest.raises(ValueError) as ve:
        cross_validate(
            est, y, error_score=None, scoring='mean_squared_error',
            cv=SlidingWindowForecastCV(window_size=100, step=24, h=1))
    assert 'error_score should be' in pytest_error_str(ve)
