# -*- coding: utf-8 -*-

"""
Tests of the ARIMA class
"""

import numpy as np
import pandas as pd

from pmdarima.arima import ARIMA, auto_arima, AutoARIMA, ARMAtoMA
from pmdarima.arima import _validation as val
from pmdarima.compat.pytest import pytest_error_str
from pmdarima.datasets import load_lynx, load_wineind, load_heartrate

from numpy.random import RandomState
from numpy.testing import assert_array_almost_equal, assert_almost_equal, \
    assert_allclose
from statsmodels import api as sm
from sklearn.metrics import mean_squared_error

import joblib
import os
import pickle
import pytest
import tempfile
import time


# initialize the random state
rs = RandomState(42)
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

hr = load_heartrate(as_series=True)
wineind = load_wineind()
lynx = load_lynx()


def test_basic_arma():
    arma = ARIMA(order=(0, 0, 0), suppress_warnings=True)
    preds = arma.fit_predict(y)  # fit/predict for coverage

    # No OOB, so assert none
    assert arma.oob_preds_ is None

    # test some of the attrs
    assert_almost_equal(arma.aic(), 11.201, decimal=3)  # equivalent in R

    # intercept is param 0
    intercept = arma.params()[0]
    assert_almost_equal(intercept, 0.441, decimal=3)  # equivalent in R
    assert_almost_equal(arma.aicc(), 11.74676, decimal=5)
    assert_almost_equal(arma.bic(), 13.639060053303311, decimal=5)

    # get predictions
    expected_preds = np.array([0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876])

    # generate predictions
    assert_array_almost_equal(preds, expected_preds)

    # Make sure we can get confidence intervals
    expected_intervals = np.array([
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139],
        [-0.10692387, 0.98852139]
    ])

    _, intervals = arma.predict(n_periods=10, return_conf_int=True,
                                alpha=0.05)
    assert_array_almost_equal(intervals, expected_intervals)


def test_issue_30():
    # From the issue:
    vec = np.array([33., 44., 58., 49., 46., 98., 97.])

    arm = AutoARIMA(out_of_sample_size=1, seasonal=False,
                    suppress_warnings=True)
    arm.fit(vec)

    # This is a way to force it:
    ARIMA(order=(0, 1, 0), out_of_sample_size=1).fit(vec)

    # Want to make sure it works with X arrays as well
    X = np.random.RandomState(1).rand(vec.shape[0], 2)
    auto_arima(vec, X=X, out_of_sample_size=1,
               seasonal=False,
               suppress_warnings=True)

    # This is a way to force it:
    ARIMA(order=(0, 1, 0), out_of_sample_size=1).fit(vec, X=X)


@pytest.mark.parametrize(
    # will be m - d
    'model', [
        ARIMA(order=(2, 0, 0)),  # arma
        ARIMA(order=(2, 1, 0)),  # arima
        ARIMA(order=(2, 1, 0), seasonal_order=(1, 0, 0, 12)),  # sarimax
    ]
)
def test_predict_in_sample_conf_int(model):
    model.fit(wineind)
    expected_m_dim = wineind.shape[0]
    preds, confints = model.predict_in_sample(return_conf_int=True, alpha=0.05)
    assert preds.shape[0] == expected_m_dim
    assert confints.shape == (expected_m_dim, 2)


@pytest.mark.parametrize(
    'model', [
        ARIMA(order=(2, 0, 0)),  # arma
        ARIMA(order=(2, 1, 0)),  # arima
        ARIMA(order=(2, 1, 0), seasonal_order=(1, 0, 0, 12)),  # sarimax
    ]
)
@pytest.mark.parametrize('X', [None, rs.rand(wineind.shape[0], 2)])
@pytest.mark.parametrize('confints', [True, False])
def test_predict_in_sample_X(model, X, confints):
    model.fit(wineind, X=X)
    res = model.predict_in_sample(X, return_conf_int=confints)
    if confints:
        assert isinstance(res, tuple) and len(res) == 2
    else:
        assert isinstance(res, np.ndarray)


def _two_times_mse(y_true, y_pred, **_):
    """A custom loss to test we can pass custom scoring metrics"""
    return mean_squared_error(y_true, y_pred) * 2


@pytest.mark.parametrize('as_pd', [True, False])
@pytest.mark.parametrize('scoring', ['mse', _two_times_mse])
def test_with_oob_and_X(as_pd, scoring):
    endog = hr
    X = np.random.RandomState(1).rand(hr.shape[0], 3)
    if as_pd:
        X = pd.DataFrame.from_records(X)
        endog = pd.Series(hr)

    arima = ARIMA(order=(2, 1, 2),
                  suppress_warnings=True,
                  scoring=scoring,
                  out_of_sample_size=10).fit(y=endog, X=X)

    # show we can get oob score and preds
    arima.oob()


def test_with_oob():
    # show we can fit with CV (kinda)
    arima = ARIMA(order=(2, 1, 2),
                  suppress_warnings=True,
                  scoring='mse',
                  out_of_sample_size=10).fit(y=hr)

    oob = arima.oob()
    assert not np.isnan(oob)  # show this works

    # Assert the predictions give the expected MAE/MSE
    oob_preds = arima.oob_preds_
    assert oob_preds.shape[0] == 10
    scoring = val.get_scoring_metric('mse')
    assert scoring(hr[-10:], oob_preds) == oob

    # show we can fit if ooss < 0 and oob will be nan
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True,
                  out_of_sample_size=-1).fit(y=hr)
    assert np.isnan(arima.oob())

    # This will raise since n_steps is not an int
    with pytest.raises(TypeError):
        arima.predict(n_periods="5")

    # But that we CAN forecast with an int...
    _ = arima.predict(n_periods=5)  # noqa: F841

    # Show we fail if cv > n_samples
    with pytest.raises(ValueError):
        ARIMA(order=(2, 1, 2), out_of_sample_size=1000).fit(hr)


# Test Issue #28 ----------------------------------------------------------
def test_oob_for_issue_28():
    # Continuation of above: can we do one with an X array, too?
    xreg = rs.rand(hr.shape[0], 4)
    arima = ARIMA(order=(2, 1, 2),
                  suppress_warnings=True,
                  out_of_sample_size=10).fit(
        y=hr, X=xreg)

    oob = arima.oob()
    assert not np.isnan(oob)

    # Assert that the endog shapes match. First is equal to the original,
    # and the second is the differenced array
    assert np.allclose(arima.arima_res_.data.endog, hr, rtol=1e-2)
    assert arima.arima_res_.model.endog.shape[0] == hr.shape[0]

    # Now assert the same for X
    assert np.allclose(arima.arima_res_.data.exog, xreg, rtol=1e-2)
    assert arima.arima_res_.model.exog.shape[0] == xreg.shape[0]

    # Compare the OOB score to an equivalent fit on data - 10 obs, but
    # without any OOB scoring, and we'll show that the OOB scoring in the
    # first IS in fact only applied to the first (train - n_out_of_bag)
    # samples
    arima_no_oob = ARIMA(
        order=(2, 1, 2), suppress_warnings=True,
        out_of_sample_size=0).fit(y=hr[:-10],
                                  X=xreg[:-10, :])

    scoring = val.get_scoring_metric(arima_no_oob.scoring)
    preds = arima_no_oob.predict(n_periods=10, X=xreg[-10:, :])
    assert np.allclose(oob, scoring(hr[-10:], preds), rtol=1e-2)

    # Show that the model parameters are not the same because the model was
    # updated.
    xreg_test = rs.rand(5, 4)
    assert not np.allclose(arima.params(), arima_no_oob.params(), rtol=1e-2)

    # Now assert on the forecast differences.
    with_oob_forecasts = arima.predict(n_periods=5, X=xreg_test)
    no_oob_forecasts = arima_no_oob.predict(n_periods=5,
                                            X=xreg_test)

    with pytest.raises(AssertionError):
        assert_array_almost_equal(with_oob_forecasts, no_oob_forecasts)

    # But after we update the no_oob model with the latest data, we should
    # be producing the same exact forecasts

    # First, show we'll fail if we try to add observations with no X
    with pytest.raises(ValueError):
        arima_no_oob.update(hr[-10:], None)

    # Also show we'll fail if we try to add mis-matched shapes of data
    with pytest.raises(ValueError):
        arima_no_oob.update(hr[-10:], xreg_test)

    # Show we fail if we try to add observations with a different dim X
    with pytest.raises(ValueError):
        arima_no_oob.update(hr[-10:], xreg_test[:, :2])

    # Actually add them now, and compare the forecasts (should be the same)
    arima_no_oob.update(hr[-10:], xreg[-10:, :])
    assert np.allclose(with_oob_forecasts,
                       arima_no_oob.predict(n_periods=5, X=xreg_test),
                       rtol=1e-2)


# Test the OOB functionality for SARIMAX (Issue #28) --------------------------

def test_oob_sarimax():
    xreg = rs.rand(wineind.shape[0], 2)
    fit = ARIMA(order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                maxiter=5,
                out_of_sample_size=15).fit(y=wineind, X=xreg)

    fit_no_oob = ARIMA(order=(1, 1, 1),
                       seasonal_order=(0, 1, 1, 12),
                       out_of_sample_size=0,
                       maxiter=5,
                       suppress_warnings=True).fit(y=wineind[:-15],
                                                   X=xreg[:-15, :])

    # now assert some of the same things here that we did in the former test
    oob = fit.oob()

    # compare scores:
    scoring = val.get_scoring_metric(fit_no_oob.scoring)
    no_oob_preds = fit_no_oob.predict(n_periods=15, X=xreg[-15:, :])
    assert np.allclose(oob, scoring(wineind[-15:], no_oob_preds), rtol=1e-2)

    # show params are no longer the same
    assert not np.allclose(fit.params(), fit_no_oob.params(), rtol=1e-2)

    # show we can add the new samples and get the exact same forecasts
    xreg_test = rs.rand(5, 2)
    fit_no_oob.update(wineind[-15:], xreg[-15:, :])
    assert np.allclose(fit.predict(5, xreg_test),
                       fit_no_oob.predict(5, xreg_test),
                       rtol=1e-2)

    # And also the params should be close now after updating
    assert np.allclose(fit.params(), fit_no_oob.params())

    # Show we can get a confidence interval out here
    preds, conf = fit.predict(5, xreg_test, return_conf_int=True)
    assert all(isinstance(a, np.ndarray) for a in (preds, conf))


# Test Issue #29 (d=0, cv=True) -----------------------------------------------


class TestIssue29:
    dta = sm.datasets.sunspots.load_pandas().data
    dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
    del dta["YEAR"]

    xreg = np.random.RandomState(1).rand(dta.shape[0], 3)

    @pytest.mark.parametrize('d', [0, 1])
    @pytest.mark.parametrize('cv', [0, 3])
    @pytest.mark.parametrize('X', [xreg, None])
    def test_oob_for_issue_29(self, d, cv, X):
        model = ARIMA(order=(2, d, 0),
                      out_of_sample_size=cv).fit(self.dta, X=X)

        # If X is defined, we need to pass n_periods of
        # X rows to the predict function. Otherwise we'll
        # just leave it at None
        if X is not None:
            xr = X[:3, :]
        else:
            xr = None

        _, _ = model.predict(n_periods=3, return_conf_int=True, X=xr)


def _try_get_attrs(arima):
    # show we can get all these attrs without getting an error
    attrs = {
        'aic', 'aicc', 'arparams', 'arroots', 'bic', 'bse', 'conf_int',
        'df_model', 'df_resid', 'hqic', 'maparams', 'maroots',
        'params', 'pvalues', 'resid',
    }

    # this just shows all of these attrs work.
    for attr in attrs:
        getattr(arima, attr)()


def test_more_elaborate():
    # show we can fit this with a non-zero order
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True).fit(y=hr)
    _try_get_attrs(arima)

    # can we fit this same arima with a made-up X array?
    xreg = rs.rand(hr.shape[0], 4)
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True).fit(y=hr, X=xreg)
    _try_get_attrs(arima)

    with tempfile.TemporaryDirectory() as tdir:

        # pickle this for the __get/setattr__ coverage.
        # since the only time this is tested is in parallel in auto.py,
        # this doesn't actually get any coverage proof...
        fl = os.path.join(tdir, 'some_temp_file.pkl')
        with open(fl, 'wb') as p:
            pickle.dump(arima, p)

        # show we can predict with this even though it's been pickled
        new_xreg = rs.rand(5, 4)
        _preds = arima.predict(n_periods=5, X=new_xreg)

        # now unpickle
        with open(fl, 'rb') as p:
            other = pickle.load(p)

        # show we can still predict, compare
        _other_preds = other.predict(n_periods=5, X=new_xreg)
        assert_array_almost_equal(_preds, _other_preds)

    # now show that since we fit the ARIMA with an X array,
    # we need to provide one for predictions otherwise it breaks.
    with pytest.raises(ValueError):
        arima.predict(n_periods=5, X=None)

    # show that if we DO provide an X and it's the wrong dims, we
    # also break things down.
    with pytest.raises(ValueError):
        arima.predict(n_periods=5, X=rs.rand(4, 4))


def test_the_r_src():
    # this is the test the R code provides
    fit = ARIMA(order=(2, 0, 1), trend='c', suppress_warnings=True).fit(abc)

    # the R code's AIC = 135.4
    assert abs(135.4 - fit.aic()) < 1.0

    # the R code's AICc = ~ 137
    assert abs(137 - fit.aicc()) < 1.0

    # the R code's BIC = ~145
    assert abs(145 - fit.bic()) < 1.0

    # R's coefficients:
    #     ar1      ar2     ma1    mean
    # -0.6515  -0.2449  0.8012  5.0370

    arparams = fit.arparams()
    assert_almost_equal(arparams, [-0.6515, -0.2449], decimal=3)

    maparams = fit.maparams()
    assert_almost_equal(maparams, [0.8012], decimal=3)

    # > fit = forecast::auto.arima(abc, max.p=5, max.d=5,
    #             max.q=5, max.order=100, stepwise=F)
    fit = auto_arima(abc, max_p=5, max_d=5, max_q=5, max_order=100,
                     seasonal=False, trend='c', suppress_warnings=True,
                     error_action='ignore')

    assert abs(135.28 - fit.aic()) < 1.0  # R's is 135.28


def test_with_seasonality():
    fit = ARIMA(order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                suppress_warnings=True).fit(y=wineind)
    _try_get_attrs(fit)

    # R code AIC result is ~3004
    assert abs(fit.aic() - 3004) < 100  # show equal within 100 or so

    # R code AICc result is ~3005
    assert abs(fit.aicc() - 3005) < 100  # show equal within 100 or so

    # R code BIC result is ~3017
    assert abs(fit.bic() - 3017) < 100  # show equal within 100 or so

    # show we can predict in-sample
    fit.predict_in_sample()

    # test with SARIMAX confidence intervals
    fit.predict(n_periods=10, return_conf_int=True, alpha=0.05)


# Test that (as of v0.9.1) we can pickle a model, pickle it again, load both
# and create predictions.
def test_double_pickle():
    arima = ARIMA(order=(0, 0, 0), trend='c', suppress_warnings=True)
    arima.fit(y)

    with tempfile.TemporaryDirectory() as tdir:

        # Now save it twice
        file_a = os.path.join(tdir, 'first.pkl')
        file_b = os.path.join(tdir, 'second.pkl')

        # No compression
        joblib.dump(arima, file_a)

        # Sleep between pickling so that the "pickle hash" for the ARIMA is
        # different by enough. We could theoretically also just use a UUID
        # for part of the hash to make sure it's unique?
        time.sleep(0.5)

        # Some compression
        joblib.dump(arima, file_b, compress=2)

        # Load both and prove they can both predict
        loaded_a = joblib.load(file_a)  # type: ARIMA
        loaded_b = joblib.load(file_b)  # type: ARIMA
        pred_a = loaded_a.predict(n_periods=5)
        pred_b = loaded_b.predict(n_periods=5)
        assert np.allclose(pred_a, pred_b)


# Regression testing for unpickling an ARIMA from an older version
def test_for_older_version():
    # Fit an ARIMA
    arima = ARIMA(order=(0, 0, 0), trend='c', suppress_warnings=True)

    # There are three possibilities here:
    # 1. The model is serialized/deserialized BEFORE it has been fit.
    #    This means we should not get a warning.
    #
    # 2. The model is saved after being fit, but it does not have a
    #    pkg_version_ attribute due to it being an old (very old) version.
    #    We still warn for this
    #
    # 3. The model is saved after the fit, and it's version does not match.
    #    We warn for this.
    for case, do_fit, expect_warning in [(1, False, False),
                                         (2, True, True),
                                         (3, True, True)]:

        # Only fit it if we should
        if do_fit:
            arima.fit(y)

        # If it's case 2, we remove the pkg_version_. If 3, we set it low
        if case == 2:
            delattr(arima, 'pkg_version_')
        elif case == 3:
            arima.pkg_version_ = '0.0.1'  # will always be < than current

        with tempfile.TemporaryDirectory() as tdir:

            pickle_file = os.path.join(tdir, 'model.pkl')
            joblib.dump(arima, pickle_file)

            # Now unpickle it and show that we get a warning (if expected)
            if expect_warning:
                with pytest.warns(UserWarning):
                    arm = joblib.load(pickle_file)  # type: ARIMA
            else:
                arm = joblib.load(pickle_file)  # type: ARIMA

            # we can still produce predictions (only if we fit)
            if do_fit:
                arm.predict(n_periods=4)


@pytest.mark.parametrize(
    'order,seasonal', [
        # ARMA
        pytest.param((1, 0, 0), (0, 0, 0, 0)),

        # ARIMA
        pytest.param((1, 1, 0), (0, 0, 0, 0)),

        # SARIMAX
        pytest.param((1, 1, 0), (1, 0, 0, 12))
    ])
def test_with_intercept(order, seasonal):
    n_params = None
    for intercept in (False, True):
        modl = ARIMA(order=order,
                     seasonal_order=seasonal,
                     with_intercept=intercept).fit(lynx)

        if not intercept:  # first time
            n_params = modl.params().shape[0]
        else:
            # With an intercept, should be 1 more
            assert modl.params().shape[0] == n_params + 1


def test_to_dict_returns_dict():
    train = lynx[:90]
    modl = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True, D=10, max_D=10,
                      error_action='ignore')
    assert isinstance(modl.to_dict(), dict)


def test_to_dict_raises_attribute_error_on_unfit_model():
    modl = ARIMA(order=(1, 1, 0))
    with pytest.raises(AttributeError):
        modl.to_dict()


# tgsmith61591: I really hate this test. But it ensures no drift, at least..
def test_to_dict_is_accurate():
    train = lynx[:90]
    modl = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                      max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                      stepwise=True, suppress_warnings=True, D=10, max_D=10,
                      error_action='ignore')
    expected = {
        'pvalues': np.array([2.04752445e-03, 1.43710465e-61,
                             1.29504002e-10, 5.22119887e-15]),
        'resid': np.array(
            [-1244.3973072, -302.89697033, -317.63342593, -304.57267897,
             131.69413491, 956.15566697, 880.37459722, 2445.86460353,
             -192.84268876, -177.1932523, -101.67727903, 384.05487582,
             -304.52047818, -570.72748088, -497.48574217, 1286.86848903,
             -400.22840217, 1017.55518758, -1157.37024626, -295.26213543,
             104.79931827, -574.9867485, -588.49652697, -535.37707505,
             -355.71298419, -164.06179682, 574.51900799, 15.45522718,
             -1358.43416826, 120.42735893, -147.94038284, -685.64124874,
             -365.18947057, -243.79704985, 317.79437422, 585.59553667,
             34.70605783, -216.21587989, -692.53375089, 116.87379358,
             -385.52193301, -540.95554558, -283.16913167, 438.72324376,
             1078.63542578, 3198.50449405, -2167.76083646, -783.80525821,
             1384.85947061, -95.84379882, -728.85293118, -35.68476597,
             211.33538732, -379.91950618, 599.42290213, -839.30599392,
             -201.97018962, -393.28468589, -376.16010796, -516.52280993,
             -369.25037143, -362.25159504, 783.17714317, 207.96692746,
             1744.27617969, -1573.37293342, -479.20751405, 473.18948601,
             -503.20223823, -648.62384466, -671.12469446, -547.51554005,
             -501.37768686, 274.76714385, 2073.1897026, -1063.19580729,
             -1664.39957997, 882.73400004, -304.17429193, -422.60267409,
             -292.34984241, -27.76090888, 1724.60937822, 3095.90133612,
             -325.78549678, 110.95150845, 645.21273504, -135.91225092,
             417.12710097, -118.27553718]),
        'order': (2, 0, 0),
        'seasonal_order': (0, 0, 0, 0),
        'oob': np.nan,
        'aic': 1487.8850037609368,
        'aicc': 1488.3555919962284,
        'bic': 1497.8842424422578,
        'bse': np.array([2.26237893e+02, 6.97744631e-02,
                         9.58556537e-02, 1.03225425e+05]),
        'params': np.array([6.97548186e+02, 1.15522102e+00,
                            -6.16136459e-01, 8.07374077e+05])
    }

    actual = modl.to_dict()

    assert actual.keys() == expected.keys()
    assert_almost_equal(actual['pvalues'], expected['pvalues'], decimal=5)
    assert_allclose(actual['resid'], expected['resid'], rtol=1e-3)
    assert actual['order'] == expected['order']
    assert actual['seasonal_order'] == expected['seasonal_order']
    assert np.isnan(actual['oob'])
    assert_almost_equal(actual['aic'], expected['aic'], decimal=5)
    assert_almost_equal(actual['aicc'], expected['aicc'], decimal=5)
    assert_almost_equal(actual['bic'], expected['bic'], decimal=5)
    assert_allclose(actual['bse'], expected['bse'], rtol=1e-3)
    assert_almost_equal(actual['params'], expected['params'], decimal=3)


def test_serialization_methods_equal():
    arima = ARIMA(order=(0, 0, 0), suppress_warnings=True).fit(y)

    with tempfile.TemporaryDirectory() as dirname:
        joblib_path = os.path.join(dirname, "joblib.pkl")
        joblib.dump(arima, joblib_path)
        loaded = joblib.load(joblib_path)
        joblib_preds = loaded.predict()

        pickle_path = os.path.join(dirname, "pickle.pkl")
        with open(pickle_path, 'wb') as p:
            pickle.dump(arima, p)

        with open(pickle_path, 'rb') as p:
            loaded = pickle.load(p)
        pickle_preds = loaded.predict()

    assert_array_almost_equal(joblib_preds, pickle_preds)


@pytest.mark.parametrize(
    'model', [
        # ARMA
        ARIMA(order=(1, 0, 0)),

        # ARIMA
        ARIMA(order=(1, 1, 2)),

        # SARIMAX
        ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
    ]
)
def test_issue_104(model):
    # Issue 104 shows that observations were not being updated appropriately.
    # We need to make sure they update for ALL models (ARMA, ARIMA, SARIMAX)
    endog = wineind
    train, test = endog[:125], endog[125:]

    model.fit(train)
    preds1 = model.predict(n_periods=100)

    model.update(test)
    preds2 = model.predict(n_periods=100)

    # These should be DIFFERENT
    assert not np.array_equal(preds1, preds2)


def test_issue_286():
    mod = ARIMA(order=(1, 1, 2))
    mod.fit(wineind)

    with pytest.raises(ValueError) as ve:
        mod.predict_in_sample(start=0)
    assert "In-sample predictions undefined for" in pytest_error_str(ve)


@pytest.mark.parametrize(
    'model', [
        # ARMA
        ARIMA(order=(1, 0, 0)),

        # ARIMA
        ARIMA(order=(1, 1, 0))
    ]
)
def test_update_1_iter(model):
    # The model should *barely* change if we update with one iter.
    endog = wineind
    train, test = endog[:145], endog[145:]

    model.fit(train)
    params1 = model.params()

    # Now update with 1 iteration, and show params have not changed too much
    model.update(test, maxiter=1)
    params2 = model.params()

    # They should be close
    assert np.allclose(params1, params2, atol=0.05)


def test_ARMAtoMA():
    ar = np.array([0.5, 0.6])
    ma = np.array([0.4, 0.3, 0.1, 0.05])
    max_deg = 6
    equivalent_ma = ARMAtoMA(ar, ma, max_deg)
    ema_expected = np.array([0.9000, 1.3500, 1.3150, 1.5175, 1.5477, 1.6843])
    assert_array_almost_equal(equivalent_ma, ema_expected, decimal=4)
