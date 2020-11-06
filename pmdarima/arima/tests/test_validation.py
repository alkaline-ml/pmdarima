# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pmdarima.compat.pytest import pytest_warning_messages, pytest_error_str
from pmdarima.arima import _validation as val
from pmdarima.warnings import ModelFitWarning


@pytest.mark.parametrize(
    'ic,ooss,expect_error,expect_warning,expected_val', [

        # happy paths
        pytest.param('aic', 0, False, False, 'aic'),
        pytest.param('aicc', 0, False, False, 'aicc'),
        pytest.param('bic', 0, False, False, 'bic'),
        pytest.param('hqic', 0, False, False, 'hqic'),
        pytest.param('oob', 10, False, False, 'oob'),

        # unhappy paths :-(
        pytest.param('aaic', 0, True, False, None),
        pytest.param('oob', 0, False, True, 'aic'),

    ]
)
def test_check_information_criterion(ic,
                                     ooss,
                                     expect_error,
                                     expect_warning,
                                     expected_val):

    if expect_error:
        with pytest.raises(ValueError) as ve:
            val.check_information_criterion(ic, ooss)
        assert 'not defined for information_criteria' in pytest_error_str(ve)

    else:
        if expect_warning:
            with pytest.warns(UserWarning) as w:
                res = val.check_information_criterion(ic, ooss)
            assert any('information_criterion cannot be' in s
                       for s in pytest_warning_messages(w))
        else:
            with pytest.warns(None) as w:
                res = val.check_information_criterion(ic, ooss)
            assert not w

        assert expected_val == res


@pytest.mark.parametrize(
    'kwargs,expected', [
        pytest.param(None, {}),
        pytest.param({}, {}),
        pytest.param({'foo': 'bar'}, {'foo': 'bar'}),
    ]
)
def test_check_kwargs(kwargs, expected):
    res = val.check_kwargs(kwargs)
    assert expected == res


@pytest.mark.parametrize(
    'm,seasonal,expect_error,expect_warning,expected_val', [

        # happy path
        pytest.param(12, True, False, False, 12),
        pytest.param(1, True, False, False, 1),
        pytest.param(0, False, False, False, 0),
        pytest.param(1, False, False, False, 0),

        # unhappy path :-(
        pytest.param(2, False, False, True, 0),
        pytest.param(0, True, True, False, None),
        pytest.param(-1, False, True, False, None),

    ]
)
def test_check_m(m, seasonal, expect_error, expect_warning, expected_val):
    if expect_error:
        with pytest.raises(ValueError) as ve:
            val.check_m(m, seasonal)
        assert 'must be a positive integer' in pytest_error_str(ve)

    else:
        if expect_warning:
            with pytest.warns(UserWarning) as w:
                res = val.check_m(m, seasonal)
            assert any('set for non-seasonal fit' in s
                       for s in pytest_warning_messages(w))
        else:
            with pytest.warns(None) as w:
                res = val.check_m(m, seasonal)
            assert not w

        assert expected_val == res


@pytest.mark.parametrize(
    'stepwise,n_jobs,expect_warning,expected_n_jobs', [

        pytest.param(False, 1, False, 1),
        pytest.param(True, 1, False, 1),
        pytest.param(False, 2, False, 2),
        pytest.param(True, 2, True, 1),

    ]
)
def test_check_n_jobs(stepwise, n_jobs, expect_warning, expected_n_jobs):
    if expect_warning:
        with pytest.warns(UserWarning) as w:
            res = val.check_n_jobs(stepwise, n_jobs)
        assert any('stepwise model cannot be fit in parallel' in s
                   for s in pytest_warning_messages(w))
    else:
        with pytest.warns(None) as w:
            res = val.check_n_jobs(stepwise, n_jobs)
        assert not w

    assert expected_n_jobs == res


@pytest.mark.parametrize(
    'st,mx,argname,exp_vals,exp_err_msg', [

        # happy paths
        pytest.param(0, 1, 'p', (0, 1), None),
        pytest.param(1, 1, 'q', (1, 1), None),
        pytest.param(1, None, 'P', (1, np.inf), None),

        # unhappy paths :-(
        pytest.param(None, 1, 'Q', None, "start_Q cannot be None"),
        pytest.param(-1, 1, 'p', None, "start_p must be positive"),
        pytest.param(2, 1, 'foo', None, "max_foo must be >= start_foo"),

    ]
)
def test_check_start_max_values(st, mx, argname, exp_vals, exp_err_msg):
    if exp_err_msg:
        with pytest.raises(ValueError) as ve:
            val.check_start_max_values(st, mx, argname)
        assert exp_err_msg in pytest_error_str(ve)
    else:
        res = val.check_start_max_values(st, mx, argname)
        assert exp_vals == res


@pytest.mark.parametrize(
    'trace,expected', [
        pytest.param(None, 0),
        pytest.param(True, 1),
        pytest.param(False, 0),
        pytest.param(1, 1),
        pytest.param(2, 2),
        pytest.param('trace it fam', 1),
        pytest.param('', 0),
    ]
)
def test_check_trace(trace, expected):
    res = val.check_trace(trace)
    assert expected == res


@pytest.mark.parametrize(
    'metric,expected_error,expected_error_msg', [
        pytest.param("mae", None, None),
        pytest.param("mse", None, None),
        pytest.param("mean_squared_error", None, None),
        pytest.param("r2_score", None, None),

        pytest.param("foo", ValueError, "is not a valid scoring"),
        pytest.param(123, TypeError, "must be a valid scoring method, or a"),
    ]
)
def test_valid_metrics(metric, expected_error, expected_error_msg):
    if not expected_error:
        assert callable(val.get_scoring_metric(metric))
    else:
        with pytest.raises(expected_error) as err:
            val.get_scoring_metric(metric)
        assert expected_error_msg in pytest_error_str(err)


@pytest.mark.parametrize(
    'd,D,expected', [
        pytest.param(0, 1, None),
        pytest.param(0, 2, "Having more than one"),
        pytest.param(2, 1, "Having 3 or more"),
        pytest.param(3, 1, "Having 3 or more"),
    ]
)
def test_warn_for_D(d, D, expected):
    if expected:
        with pytest.warns(ModelFitWarning) as mfw:
            val.warn_for_D(d=d, D=D)

            warning_msgs = pytest_warning_messages(mfw)
            assert any(expected in w for w in warning_msgs)

    else:
        with pytest.warns(None):
            val.warn_for_D(d=d, D=D)
