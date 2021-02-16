# -*- coding: utf-8 -*-

from pmdarima.compat.pytest import pytest_error_str
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


def test_bad_window_size():
    cv = SlidingWindowForecastCV(window_size=2, step=1, h=4)
    with pytest.raises(ValueError) as ve:
        list(cv.split(y))
    assert "> 2" in pytest_error_str(ve)


def test_issue_364_bad_splits():
    endog = y[:100]
    cv = SlidingWindowForecastCV(window_size=90, step=1, h=4)
    gen = cv.split(endog)

    expected = [
        (np.arange(0, 90), np.array([90, 91, 92, 93])),
        (np.arange(1, 91), np.array([91, 92, 93, 94])),
        (np.arange(2, 92), np.array([92, 93, 94, 95])),
        (np.arange(3, 93), np.array([93, 94, 95, 96])),
        (np.arange(4, 94), np.array([94, 95, 96, 97])),
        (np.arange(5, 95), np.array([95, 96, 97, 98])),
        (np.arange(6, 96), np.array([96, 97, 98, 99])),
    ]

    # should be 7
    for i, (train, test) in enumerate(gen):
        assert_array_equal(train, expected[i][0])
        assert_array_equal(test, expected[i][1])

    # assert no extra splits
    with pytest.raises(StopIteration):
        next(gen)


def test_rolling_forecast_cv_bad_splits():
    endog = y[:100]
    cv = RollingForecastCV(initial=90, step=1, h=4)
    gen = cv.split(endog)

    expected = [
        (np.arange(0, 90), np.array([90, 91, 92, 93])),
        (np.arange(0, 91), np.array([91, 92, 93, 94])),
        (np.arange(0, 92), np.array([92, 93, 94, 95])),
        (np.arange(0, 93), np.array([93, 94, 95, 96])),
        (np.arange(0, 94), np.array([94, 95, 96, 97])),
        (np.arange(0, 95), np.array([95, 96, 97, 98])),
        (np.arange(0, 96), np.array([96, 97, 98, 99])),
    ]

    # should be 7
    for i, (train, test) in enumerate(gen):
        assert_array_equal(train, expected[i][0])
        assert_array_equal(test, expected[i][1])

    # assert no extra splits
    with pytest.raises(StopIteration):
        next(gen)
