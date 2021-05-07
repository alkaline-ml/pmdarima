# -*- coding: utf-8 -*-

from pmdarima.arima import ARIMA
from pmdarima.warnings import ModelFitWarning
from pmdarima.compat.pytest import pytest_error_str
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima.model_selection._split import RollingForecastCV, \
    SlidingWindowForecastCV
from pmdarima.model_selection._validation import cross_val_score, \
    _check_scoring, cross_validate, cross_val_predict, _check_averaging
from pmdarima.datasets import load_airpassengers
import pytest
import numpy as np
from unittest import mock

y = load_airpassengers()
exogenous = np.random.RandomState(1).rand(y.shape[0], 2)


@pytest.mark.parametrize('cv', [
    SlidingWindowForecastCV(window_size=100, step=24, h=1),
    RollingForecastCV(initial=120, step=12, h=1),
])
@pytest.mark.parametrize(
    'est', [
        ARIMA(order=(2, 1, 1), maxiter=2, simple_differencing=True),
        ARIMA(order=(1, 1, 2),
              seasonal_order=(0, 1, 1, 12),
              maxiter=2,
              simple_differencing=True,
              suppress_warnings=True),
        Pipeline([
            ("fourier", FourierFeaturizer(m=12)),
            ("arima", ARIMA(order=(2, 1, 0),
                            maxiter=2,
                            simple_differencing=True))
        ])
    ]
)
@pytest.mark.parametrize('verbose', [0, 2, 4])
@pytest.mark.parametrize('X', [None, exogenous])
def test_cv_scores(cv, est, verbose, X):
    scores = cross_val_score(
        est, y, X=X, scoring='mean_squared_error',
        cv=cv, verbose=verbose)
    assert isinstance(scores, np.ndarray)


@pytest.mark.parametrize('cv', [
    SlidingWindowForecastCV(window_size=100, step=12, h=12),
    RollingForecastCV(initial=120, step=12, h=12),
])
@pytest.mark.parametrize(
    'est', [
        ARIMA(order=(2, 1, 1), simple_differencing=True),
        ARIMA(order=(1, 1, 2),
              seasonal_order=(0, 1, 1, 12),
              simple_differencing=True,
              suppress_warnings=True),
        Pipeline([
            ("fourier", FourierFeaturizer(m=12)),
            ("arima", ARIMA(order=(2, 1, 0),
                            maxiter=2,
                            simple_differencing=True))
        ])
    ]
)
@pytest.mark.parametrize('avg', ["mean", "median"])
@pytest.mark.parametrize('return_raw_predictions', [True, False])
def test_cv_predictions(cv, est, avg, return_raw_predictions):
    preds = cross_val_predict(
        est, y, cv=cv, verbose=4, averaging=avg,
        return_raw_predictions=return_raw_predictions)
    assert isinstance(preds, np.ndarray)
    if return_raw_predictions:
        assert preds.shape[0] == len(y)
        assert preds.shape[1] == cv.horizon
    else:
        assert preds.ndim == 1


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


def test_check_averaging():
    # This will work since it's a callable
    avg = (lambda x, axis: x)
    assert _check_averaging(avg) is avg

    # fails for bad method
    with pytest.raises(ValueError):
        _check_averaging('bad method')

    # fails for anything else
    with pytest.raises(TypeError):
        _check_averaging(123)


def test_cross_val_predict_error():
    cv = SlidingWindowForecastCV(step=24, h=1)
    with pytest.raises(ValueError):
        cross_val_predict(ARIMA(order=(2, 1, 0), maxiter=3), y, cv=cv)


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
