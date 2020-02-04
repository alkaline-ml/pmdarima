# -*- coding: utf-8 -*-

from pmdarima.model_selection import RollingForecastCV, \
    SlidingWindowForecastCV, check_cv, train_test_split
from pmdarima.datasets import load_wineind
import pytest
import numpy as np
from numpy.testing import assert_array_equal

y = load_wineind()


@pytest.mark.parametrize(
    'cv', [
        RollingForecastCV(),
        RollingForecastCV(h=4),
        RollingForecastCV(initial=150, h=10),
        RollingForecastCV(initial=12, h=16, step=7),
    ]
)
def test_rolling_forecast_cv_passing(cv):
    # get all splits
    splits = list(cv.split(y))
    last_train_step = None
    for train, test in splits:
        assert test.shape[0] == cv.h
        assert test[-1] == train[-1] + cv.h

        if last_train_step is not None:
            assert train[-1] == last_train_step + cv.step
        last_train_step = train[-1]


@pytest.mark.parametrize(
    'cv', [
        SlidingWindowForecastCV(),
        SlidingWindowForecastCV(h=4),
        SlidingWindowForecastCV(window_size=42, h=10),
        SlidingWindowForecastCV(window_size=67, h=16, step=7),
    ]
)
def test_sliding_forecast_cv_passing(cv):
    # get all splits
    splits = list(cv.split(y))
    last_train_step = None
    last_window_size = None
    for train, test in splits:
        assert test.shape[0] == cv.h
        assert test[-1] == train[-1] + cv.h

        if last_train_step is not None:
            assert train[-1] == last_train_step + cv.step
        last_train_step = train[-1]

        if last_window_size is not None:
            assert train.shape[0] == last_window_size
        last_window_size = train.shape[0]

        # only assert this if it's defined in the constructor
        if cv.window_size:
            assert cv.window_size == train.shape[0]


@pytest.mark.parametrize(
    'cv', [
        RollingForecastCV(initial=-1),  # too low initial
        RollingForecastCV(initial=150, h=100),  # too high sum of initial + h
        SlidingWindowForecastCV(window_size=500),  # too high window
    ]
)
def test_cv_split_value_errors(cv):
    with pytest.raises(ValueError):
        list(cv.split(y))


def test_cv_constructor_value_errors():
    with pytest.raises(ValueError):
        RollingForecastCV(h=-1),  # too low horizon

    with pytest.raises(ValueError):
        RollingForecastCV(step=-1),  # too low step


def test_check_cv():
    cv = SlidingWindowForecastCV(h=12)
    assert check_cv(cv) is cv
    assert isinstance(check_cv(None), RollingForecastCV)

    with pytest.raises(TypeError):
        check_cv('something else')


def test_train_test_split():
    tr, te = train_test_split(y, test_size=10)
    assert te.shape[0] == 10
    assert_array_equal(y, np.concatenate([tr, te]))
