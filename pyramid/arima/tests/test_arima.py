
from __future__ import absolute_import
from pyramid.arima import ARIMA, auto_arima
from pyramid.arima.auto import _fmt_warning_str
from pyramid.arima.utils import nsdiffs
from pyramid.datasets import load_lynx, load_wineind
from nose.tools import assert_raises
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from numpy.random import RandomState
import warnings
import pickle
import os

# initialize the random state
rs = RandomState(42)
y = rs.rand(25)

# more interesting, heart rate data:
hr = np.array([84.2697, 84.2697, 84.0619, 85.6542, 87.2093, 87.1246,
               86.8726, 86.7052, 87.5899, 89.1475, 89.8204, 89.8204,
               90.4375, 91.7605, 93.1081, 94.3291, 95.8003, 97.5119,
               98.7457, 98.904, 98.3437, 98.3075, 98.8313, 99.0789,
               98.8157, 98.2998, 97.7311, 97.6471, 97.7922, 97.2974,
               96.2042, 95.2318, 94.9367, 95.0867, 95.389, 95.5414,
               95.2439, 94.9415, 95.3557, 96.3423, 97.1563, 97.4026,
               96.7028, 96.5516, 97.9837, 98.9879, 97.6312, 95.4064,
               93.8603, 93.0552, 94.6012, 95.8476, 95.7692, 95.9236,
               95.7692, 95.9211, 95.8501, 94.6703, 93.0993, 91.972,
               91.7821, 91.7911, 90.807, 89.3196, 88.1511, 88.7762,
               90.2265, 90.8066, 91.2284, 92.4238, 93.243, 92.8472,
               92.5926, 91.7778, 91.2974, 91.6364, 91.2952, 91.771,
               93.2285, 93.3199, 91.8799, 91.2239, 92.4055, 93.8716,
               94.5825, 94.5594, 94.9453, 96.2412, 96.6879, 95.8295,
               94.7819, 93.4731, 92.7997, 92.963, 92.6996, 91.9648,
               91.2417, 91.9312, 93.9548, 95.3044, 95.2511, 94.5358,
               93.8093, 93.2287, 92.2065, 92.1588, 93.6376, 94.899,
               95.1592, 95.2415, 95.5414, 95.0971, 94.528, 95.5887,
               96.4715, 96.6158, 97.0769, 96.8531, 96.3947, 97.4291,
               98.1767, 97.0148, 96.044, 95.9581, 96.4814, 96.5211,
               95.3629, 93.5741, 92.077, 90.4094, 90.1751, 91.3312,
               91.2883, 89.0592, 87.052, 86.6226, 85.7889, 85.6348,
               85.3911, 83.8064, 82.8729, 82.6266, 82.645, 82.645,
               82.645, 82.645, 82.645, 82.645, 82.645, 82.645])

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

wineind = load_wineind()
lynx = load_lynx()


def test_basic_arima():
    arima = ARIMA(order=(0, 0, 0), trend='c', suppress_warnings=True)
    preds = arima.fit_predict(y)  # fit/predict for coverage

    # test some of the attrs
    assert_almost_equal(arima.aic(), 11.201308403566909, decimal=5)
    assert_almost_equal(arima.aicc(), 11.74676, decimal=5)
    assert_almost_equal(arima.bic(), 13.639060053303311, decimal=5)

    # get predictions
    expected_preds = np.array([0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876])

    # generate predictions
    assert_array_almost_equal(preds, expected_preds)


def test_with_oob():
    # show we can fit with CV (kinda)
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True,
                  out_of_sample_size=10).fit(y=hr)
    assert not np.isnan(arima.oob())  # show this works

    # show we can fit if ooss < 0 and oob will be nan
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True,
                  out_of_sample_size=-1).fit(y=hr)
    assert np.isnan(arima.oob())

    # can we do one with an exogenous array, too?
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True,
                  out_of_sample_size=10).fit(
        y=hr, exogenous=rs.rand(hr.shape[0], 4))
    assert not np.isnan(arima.oob())


def _try_get_attrs(arima):
    # show we can get all these attrs without getting an error
    attrs = {
        'aic', 'aicc', 'arparams', 'arroots', 'bic', 'bse',
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

    # can we fit this same arima with a made-up exogenous array?
    xreg = rs.rand(hr.shape[0], 4)
    arima = ARIMA(order=(2, 1, 2), suppress_warnings=True).fit(
        y=hr, exogenous=xreg)
    _try_get_attrs(arima)

    # pickle this for the __get/setattr__ coverage.
    # since the only time this is tested is in parallel in auto.py,
    # this doesn't actually get any coverage proof...
    fl = 'some_temp_file.pkl'
    with open(fl, 'wb') as p:
        pickle.dump(arima, p)

    # show we can predict with this even though it's been pickled
    new_xreg = rs.rand(5, 4)
    _preds = arima.predict(n_periods=5, exogenous=new_xreg)

    # now unpickle
    with open(fl, 'rb') as p:
        other = pickle.load(p)

    # show we can still predict, compare
    _other_preds = other.predict(n_periods=5, exogenous=new_xreg)
    assert_array_almost_equal(_preds, _other_preds)

    # now clear the cache and remove the pickle file
    arima._clear_cached_state()
    os.unlink(fl)

    # now show that since we fit the ARIMA with an exogenous array,
    # we need to provide one for predictions otherwise it breaks.
    assert_raises(ValueError, arima.predict, n_periods=5, exogenous=None)

    # show that if we DO provide an exogenous and it's the wrong dims, we
    # also break things down.
    assert_raises(ValueError, arima.predict, n_periods=5,
                  exogenous=rs.rand(4, 4))


def test_the_r_src():
    # this is the test the R code provides
    fit = ARIMA(order=(2, 0, 1), trend='c', suppress_warnings=True).fit(abc)

    # the R code's AIC = ~135
    assert abs(135 - fit.aic()) < 1.0

    # the R code's AICc = ~ 137
    assert abs(137 - fit.aicc()) < 1.0

    # the R code's BIC = ~145
    assert abs(145 - fit.bic()) < 1.0

    # R's coefficients:
    #     ar1      ar2     ma1    mean
    # -0.6515  -0.2449  0.8012  5.0370

    # note that statsmodels' mean is on the front, not the end.
    params = fit.params()
    assert_almost_equal(params, np.array([5.0370, -0.6515, -0.2449, 0.8012]),
                        decimal=2)

    # > fit = forecast::auto.arima(abc, max.p=5, max.d=5,
    #             max.q=5, max.order=100, stepwise=F)
    fit = auto_arima(abc, max_p=5, max_d=5, max_q=5, max_order=100,
                     seasonal=False, trend='c', suppress_warnings=True,
                     error_action='ignore')

    # this differs from the R fit with a slightly higher AIC...
    assert abs(137 - fit.aic()) < 1.0  # R's is 135.28


def test_errors():
    def _assert_val_error(f, *args, **kwargs):
        try:
            f(*args, **kwargs)
            return False
        except ValueError:
            return True

    # show we fail for bad start/max p, q values:
    _assert_val_error(auto_arima, abc, start_p=-1)
    _assert_val_error(auto_arima, abc, start_q=-1)
    _assert_val_error(auto_arima, abc, max_p=-1)
    _assert_val_error(auto_arima, abc, max_q=-1)
    _assert_val_error(auto_arima, abc, start_p=0, max_p=0)
    _assert_val_error(auto_arima, abc, start_q=0, max_q=0)

    # show max order error
    _assert_val_error(auto_arima, abc, max_order=-1)

    # show errors for d
    _assert_val_error(auto_arima, abc, max_d=-1)
    _assert_val_error(auto_arima, abc, d=-1)
    _assert_val_error(auto_arima, abc, d=5, max_d=4)

    # show error for bad IC
    _assert_val_error(auto_arima, abc, information_criterion='bad-value')

    # show bad m value
    _assert_val_error(auto_arima, abc, m=0)

    # show that for starting values > max_order, we'll get an error
    _assert_val_error(auto_arima, abc, start_p=5, start_q=5,
                      seasonal=False, max_order=3)
    _assert_val_error(auto_arima, abc, start_p=5, start_q=5, start_P=4,
                      start_Q=3, seasonal=True, max_order=3)


def test_many_orders():
    lam = 0.5
    lynx_bc = ((lynx ** lam) - 1) / lam
    auto_arima(lynx_bc, start_p=1, start_q=1, d=0, max_p=5, max_q=5,
               suppress_warnings=True, stepwise=True)


def test_with_seasonality1():
    fit = ARIMA(order=(1, 1, 1),
                seasonal_order=(0, 1, 1, 12),
                suppress_warnings=True).fit(y=wineind)
    _try_get_attrs(fit)

    # R code AIC result is ~3004
    assert abs(fit.aic() - 3004) < 100  # show equal within 100 or so

    # R code AICc result is ~3005
    assert abs(fit.aicc() - 3005) < 100 # show equal within 100 or so

    # R code BIC result is ~3017
    assert abs(fit.bic() - 3017) < 100  # show equal within 100 or so

    # show we can predict in-sample
    fit.predict_in_sample()


def test_with_seasonality2():
    # also test the warning, while we're at it...
    def suppress_warnings(func):
        def suppressor(*args, **kwargs):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        return suppressor

    @suppress_warnings
    def do_fit():
        return auto_arima(wineind, start_p=1, start_q=1, max_p=2,
                          max_q=2, m=12, start_P=0, seasonal=True, n_jobs=-1,
                          d=1, D=1, stepwise=True,
                          suppress_warnings=True,
                          error_action='ignore',
                          random_state=42)

    # show that we can forecast even after the
    # pickling (this was fit in parallel)
    seasonal_fit = do_fit()
    seasonal_fit.predict(n_periods=10)

    # ensure summary still works
    seasonal_fit.summary()


def test_with_seasonality3():
    # show we can estimate D even when it's not there...
    auto_arima(wineind, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
               start_P=0, seasonal=True, d=1, D=None,
               error_action='ignore', suppress_warnings=True,
               trace=True,  # get the coverage on trace
               random_state=42, stepwise=True)


def test_with_seasonality4():
    # show we can run a random search much faster! and while we're at it,
    # make the function return all the values.
    auto_arima(wineind, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
               start_P=0, seasonal=True, n_jobs=1, d=1, D=None, stepwise=False,
               error_action='ignore', suppress_warnings=True,
               random=True, random_state=42, return_valid_fits=True,
               n_fits=5)  # only fit 5


def test_with_seasonality5():
    # can we fit the same thing with an exogenous array of predictors?
    # also make it stationary and make sure that works...
    all_res = auto_arima(wineind, start_p=1, start_q=1, max_p=2,
                         max_q=2, m=12, start_P=0, seasonal=True, n_jobs=1,
                         d=1, D=None, error_action='ignore',
                         suppress_warnings=True, stationary=True,
                         random=True, random_state=42, return_valid_fits=True,
                         stepwise=False, n_fits=5,
                         exogenous=rs.rand(wineind.shape[0], 4))  # only fit 2

    # show it is a list
    assert hasattr(all_res, '__iter__')


def test_with_seasonality6():
    # show that we can fit an ARIMA where the max_p|q == start_p|q
    auto_arima(hr, start_p=0, max_p=0, d=0, start_q=0, max_q=0,
               seasonal=False, max_order=np.inf,
               suppress_warnings=True)

    # FIXME: we get an IndexError from statsmodels summary if (0, 0, 0)


def test_with_seasonality7():
    # show we can fit one with OOB as the criterion
    auto_arima(wineind, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
               start_P=0, seasonal=True, n_jobs=1, d=1, D=1,
               out_of_sample_size=10, information_criterion='oob',
               suppress_warnings=True,
               error_action='raise',  # do raise so it fails fast
               random=True, random_state=42, n_fits=3,
               stepwise=False)


def test_corner_cases():
    assert_raises(ValueError, auto_arima, wineind,
                  error_action='some-bad-string')

    # things that produce warnings
    with warnings.catch_warnings(record=False):
        warnings.simplefilter('ignore')

        # show a constant result will result in a quick fit
        auto_arima(np.ones(10), suppress_warnings=True)

        # show the same thing with return_all results in the ARIMA in a list
        fits = auto_arima(np.ones(10), suppress_warnings=True,
                          return_valid_fits=True)
        assert hasattr(fits, '__iter__')

    # show we fail for n_iter < 0
    assert_raises(ValueError, auto_arima, np.ones(10), random=True, n_fits=-1)

    # show if max* < start* it breaks:
    assert_raises(ValueError, auto_arima, np.ones(10), start_p=5, max_p=0)


def test_warning_str_fmt():
    order = (1, 1, 1)
    seasonal = (1, 1, 1, 1)
    for ssnl in (seasonal, None):
        _fmt_warning_str(order, ssnl)


def test_nsdiffs_on_wine():
    assert nsdiffs(wineind, m=52) == 2
