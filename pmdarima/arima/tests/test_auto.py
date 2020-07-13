# -*- coding: utf-8 -*-

"""
Tests of auto-arima function and class
"""

import numpy as np
import pandas as pd

import pmdarima as pm
from pmdarima.arima import auto
from pmdarima.arima.utils import nsdiffs
from pmdarima.arima.warnings import ModelFitWarning
from pmdarima.compat.pytest import pytest_error_str, pytest_warning_messages

from numpy.testing import assert_array_almost_equal

import os
from os.path import abspath, dirname
import pytest

# initialize the random state
rs = np.random.RandomState(42)
y = rs.rand(25)

# > set.seed(123)
# > abc <- rnorm(50, 5, 1)
abc = np.array([4.439524, 4.769823, 6.558708, 5.070508,
                5.129288, 6.715065, 5.460916, 3.734939,
                4.313147, 4.554338, 6.224082, 5.359814,
                5.400771, 5.110683, 4.444159, 6.786913,
                5.497850, 3.033383, 5.701356, 4.527209,
                3.932176, 4.782025, 3.973996, 4.271109,
                4.374961, 3.313307, 5.837787, 5.153373,
                3.861863, 6.253815, 5.426464, 4.704929,
                5.895126, 5.878133, 5.821581, 5.688640,
                5.553918, 4.938088, 4.694037, 4.619529,
                4.305293, 4.792083, 3.734604, 7.168956,
                6.207962, 3.876891, 4.597115, 4.533345,
                5.779965, 4.916631])

airpassengers = pm.datasets.load_airpassengers()
austres = pm.datasets.load_austres()
hr = pm.datasets.load_heartrate(as_series=True)
lynx = pm.datasets.load_lynx()
wineind = pm.datasets.load_wineind()

# A random xreg for the wineind array
wineind_xreg = rs.rand(wineind.shape[0], 2)

# Yes, m is ACTUALLY 12... but that takes a LONG time. If we set it to
# 1, we actually get a much, much faster model fit. We can only use this
# if we're NOT testing the output of the model, but just the functionality!
wineind_m = 1


def test_AutoARIMA_class():
    train, test = wineind[:125], wineind[125:]
    mod = pm.AutoARIMA(maxiter=5)
    mod.fit(train)

    endog = mod.model_.arima_res_.data.endog
    assert_array_almost_equal(train, endog)

    # update
    mod.update(test, maxiter=2)
    new_endog = mod.model_.arima_res_.data.endog
    assert_array_almost_equal(wineind, new_endog)


def test_corner_cases():
    with pytest.raises(ValueError):
        pm.auto_arima(wineind, error_action='some-bad-string')

    # things that produce warnings
    with pytest.warns(UserWarning):
        # show a constant result will result in a quick fit
        pm.auto_arima(np.ones(10), suppress_warnings=True)

        # show the same thing with return_all results in the ARIMA in a list
        fits = pm.auto_arima(np.ones(10), suppress_warnings=True,
                             return_valid_fits=True)
        assert hasattr(fits, '__iter__')

    # show we fail for n_iter < 0
    with pytest.raises(ValueError):
        pm.auto_arima(np.ones(10), random=True, n_fits=-1)

    # show if max* < start* it breaks:
    with pytest.raises(ValueError):
        pm.auto_arima(np.ones(10), start_p=5, max_p=0)


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


# Force case where data is simple polynomial after differencing
@pytest.mark.filterwarnings('ignore:divide by zero')  # Expected, so ignore
def test_force_polynomial_error():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    d = 3
    xreg = None

    with pytest.raises(ValueError) as ve, \
            pytest.warns(ModelFitWarning) as mfw:
        pm.auto_arima(x, d=d, D=0, seasonal=False, exogenous=xreg, trace=2)

    err_msg = pytest_error_str(ve)
    assert 'simple polynomial' in err_msg, err_msg

    warning_msgs = pytest_warning_messages(mfw)
    assert any('more differencing operation' in w for w in warning_msgs)


# Show that we can complete when max order is None
def test_inf_max_order():
    _ = pm.auto_arima(lynx, max_order=None,  # noqa: F841
                      suppress_warnings=True,
                      error_action='trace')


# "ValueError: negative dimensions are not allowed" in OCSB test
def test_issue_191():
    X = pd.read_csv(
        os.path.join(abspath(dirname(__file__)), 'data', 'issue_191.csv'))
    y = X[X.columns[1]].values
    pm.auto_arima(
        y,
        error_action="warn",
        seasonal=True,
        m=12,
        alpha=0.05,
        suppress_warnings=True,
        trace=True)


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


# Asserting where D grows too large as a product of an M that's too big.
def test_m_too_large():
    train = lynx[:90]

    with pytest.raises(ValueError) as v:
        pm.auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True, D=10, max_D=10,
                      error_action='ignore', m=20)

    msg = pytest_error_str(v)
    assert 'The seasonal differencing order' in msg


def test_many_orders():
    lam = 0.5
    lynx_bc = ((lynx ** lam) - 1) / lam
    pm.auto_arima(lynx_bc, start_p=1, start_q=1, d=0, max_p=5, max_q=5,
                  suppress_warnings=True, stepwise=True)


@pytest.mark.parametrize(
    'data,test,m,expected', [
        pytest.param(wineind, 'ch', 52, 2),
        pytest.param(wineind, 'ch', 12, 0),
        pytest.param(wineind, 'ocsb', 52, 0),
        pytest.param(austres, 'ocsb', 4, 0)
    ]
)
def test_nsdiffs_on_various(data, test, m, expected):
    assert nsdiffs(data, m=m, test=test, max_D=3) == expected


def test_oob_with_zero_out_of_sample_size():
    with pytest.warns(UserWarning) as uw:
        pm.auto_arima(y, suppress_warnings=False, information_criterion="oob",
                      out_of_sample_size=0)

    assert uw[0].message.args[0] == "information_criterion cannot be 'oob' " \
                                    "with out_of_sample_size = 0. Falling " \
                                    "back to information criterion = aic."


@pytest.mark.parametrize(
    'dataset,m,kwargs,expected_order,expected_seasonal', [

        # model <- auto.arima(AirPassengers, trace=TRUE)
        pytest.param(
            airpassengers, 12, {}, (2, 1, 1), (0, 1, 0),
        ),

        # TODO: eventually some more.
    ]
)
def test_r_equivalency(dataset, m, kwargs, expected_order, expected_seasonal):
    fit = pm.auto_arima(dataset, m=m, trace=1, suppress_warnings=True)
    assert fit.order == expected_order
    assert fit.seasonal_order[:3] == expected_seasonal


@pytest.mark.parametrize('endog', [austres, pd.Series(austres)])
def test_random_with_oob(endog):
    # show we can fit one with OOB as the criterion
    pm.auto_arima(endog, start_p=1, start_q=1, max_p=2, max_q=2, m=4,
                  start_P=0, seasonal=True, n_jobs=1, d=1, D=1,
                  out_of_sample_size=10, information_criterion='oob',
                  suppress_warnings=True,
                  error_action='raise',  # do raise so it fails fast
                  random=True, random_state=42, n_fits=2,
                  stepwise=False,

                  # Set to super low iter to make test move quickly
                  maxiter=3)


# Test if exogenous is not None and D > 0
@pytest.mark.parametrize('m', [2])  # , 12])
def test_seasonal_xreg_differencing(m):
    # Test both a small M and a large M since M is used as the lag parameter
    # in the xreg array differencing. If M is 1, D is set to 0
    _ = pm.auto_arima(wineind, d=1, D=1,  # noqa: F841
                      seasonal=True,
                      exogenous=wineind_xreg, error_action='ignore',
                      suppress_warnings=True, m=m,

                      # Set to super low iter to make test move quickly
                      maxiter=5)


def test_small_samples():
    # if n_samples < 10, test the new starting p, d, Q
    samp = lynx[:8]
    pm.auto_arima(samp, suppress_warnings=True, stepwise=True,
                  error_action='ignore')


def test_start_pq_equal_max_pq():
    # show that we can fit an ARIMA where the max_p|q == start_p|q
    m = pm.auto_arima(hr, start_p=0, max_p=0, d=0, start_q=0, max_q=0,
                      seasonal=False, max_order=np.inf,
                      suppress_warnings=True)

    # older versions of sm would raise IndexError for (0, 0, 0) on summary
    m.summary()


@pytest.mark.parametrize(
    'endog, max_order, kwargs', [
        # show that for starting values > max_order, we can still get a fit
        pytest.param(abc, 3, {'start_p': 5,
                              'start_q': 5,
                              'seasonal': False,
                              'stepwise': False}),

        pytest.param(abc, 3, {'start_p': 5,
                              'start_q': 5,
                              'start_P': 2,
                              'start_Q': 2,
                              'seasonal': True,
                              'stepwise': False}),
    ]
)
def test_valid_max_order_edges(endog, max_order, kwargs):
    fit = pm.auto_arima(endog, max_order=max_order, **kwargs)
    order = fit.order
    ssnal = fit.seasonal_order
    assert (sum(order) + sum(ssnal[:3])) <= max_order


@pytest.mark.parametrize(
    'endog, kwargs', [
        # other assertions
        pytest.param(abc, {'max_order': -1, 'stepwise': False}),
        pytest.param(abc, {'max_d': -1}),
        pytest.param(abc, {'d': -1}),
        pytest.param(abc, {'max_D': -1}),
        pytest.param(abc, {'D': -1}),
    ]
)
def test_value_errors(endog, kwargs):
    with pytest.raises(ValueError):
        pm.auto_arima(endog, **kwargs)


def test_warn_for_large_differences():
    # First: d is too large
    with pytest.warns(ModelFitWarning) as w:
        pm.auto_arima(wineind, seasonal=True, m=1, suppress_warnings=False,
                      d=3, maxiter=5)
    assert any('Having 3 or more differencing operations' in s
               for s in pytest_warning_messages(w))

    # Second: D is too large. M needs to be > 1 or D will be set to 0...
    # unfortunately, this takes a long time.
    with pytest.warns(ModelFitWarning) as w:
        pm.auto_arima(wineind, seasonal=True, m=2,  # noqa: F841
                      suppress_warnings=False,
                      D=3,
                      maxiter=5)
    assert any('Having more than one seasonal differences' in s
               for s in pytest_warning_messages(w))


def test_stepwise_with_simple_differencing():
    def do_fit(simple_differencing):
        return pm.auto_arima(wineind, start_p=1, start_q=1, max_p=2,
                             max_q=2, m=2, start_P=0,
                             seasonal=True,
                             d=1, D=1, stepwise=True,
                             error_action='ignore',
                             sarimax_kwargs={
                                 'simple_differencing': simple_differencing
                             },
                             maxiter=2)

    # show that we can forecast even after the
    # pickling (this was fit in parallel)
    seasonal_fit = do_fit(False)
    seasonal_fit.predict(n_periods=10)

    # ensure summary still works
    seasonal_fit.summary()

    # Show we can predict on seasonal where conf_int is true
    seasonal_fit.predict(n_periods=10, return_conf_int=True)

    # We should get the same order when simple_differencing
    simple = do_fit(True)
    assert simple.order == seasonal_fit.order
    assert simple.seasonal_order == seasonal_fit.seasonal_order


def test_with_seasonality2():
    # show we can estimate D even when it's not there...
    pm.auto_arima(wineind, start_p=1, start_q=1, max_p=2, max_q=2, m=wineind_m,
                  start_P=0, seasonal=True, d=1, D=None,
                  error_action='ignore', suppress_warnings=True,
                  trace=True,  # get the coverage on trace
                  random_state=42, stepwise=True,

                  # Set to super low iter to make test move quickly
                  maxiter=5)


def test_with_seasonality3():
    # show we can run a random search much faster! and while we're at it,
    # make the function return all the values. Also, use small M to make our
    # lives easier.
    pm.auto_arima(wineind, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
                  start_P=0, seasonal=True, n_jobs=1, d=1, D=None,
                  stepwise=False, error_action='ignore',
                  suppress_warnings=True, random=True, random_state=42,
                  return_valid_fits=True,
                  n_fits=3,  # only a few

                  # Set to super low iter to make test move quickly
                  maxiter=5)


def test_with_seasonality4():
    # can we fit the same thing with an exogenous array of predictors?
    # also make it stationary and make sure that works...
    # 9/22/18 - make not parallel to reduce mem overhead on pytest
    all_res = pm.auto_arima(wineind, start_p=1, start_q=1, max_p=2,
                            max_q=2, m=12, start_P=0, seasonal=True,
                            d=1, D=None, error_action='ignore',
                            suppress_warnings=True, stationary=True,
                            random_state=42, return_valid_fits=True,
                            stepwise=True,
                            exogenous=rs.rand(wineind.shape[0], 4),

                            # Set to super low iter to make test move quickly
                            maxiter=5)

    # show it is a list
    assert hasattr(all_res, '__iter__')
