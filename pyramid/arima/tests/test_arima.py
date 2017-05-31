
from __future__ import absolute_import
from pyramid.arima import ARIMA, auto_arima
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from numpy.random import RandomState

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
abc = np.array([4.439524, 4.769823, 6.558708, 5.070508, 5.129288, 6.715065, 5.460916, 3.734939,
                4.313147, 4.554338, 6.224082, 5.359814, 5.400771, 5.110683, 4.444159, 6.786913,
                5.497850, 3.033383, 5.701356, 4.527209, 3.932176, 4.782025, 3.973996, 4.271109,
                4.374961, 3.313307, 5.837787, 5.153373, 3.861863, 6.253815, 5.426464, 4.704929,
                5.895126, 5.878133, 5.821581, 5.688640, 5.553918, 4.938088, 4.694037, 4.619529,
                4.305293, 4.792083, 3.734604, 7.168956, 6.207962, 3.876891, 4.597115, 4.533345,
                5.779965, 4.916631])


def test_basic_arima():
    arima = ARIMA(order=(0, 0, 0)).fit(y)

    # test some of the attrs
    assert_almost_equal(arima.aic(), 11.201308403566909, decimal=5)
    assert_almost_equal(arima.bic(), 13.639060053303311, decimal=5)

    # get predictions
    expected_preds = np.array([0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876])

    expected_stderr = np.array([0.27945546, 0.27945546, 0.27945546,
                                0.27945546, 0.27945546, 0.27945546,
                                0.27945546, 0.27945546, 0.27945546,
                                0.27945546])

    expected_conf_int = np.array([[-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139],
                                  [-0.10692387,  0.98852139]])

    # generate predictions with everything in output
    preds, stderr, conf_int = arima.predict(n_periods=10, include_conf_int=True, include_std_err=True)
    assert_array_almost_equal(preds, expected_preds)
    assert_array_almost_equal(stderr, expected_stderr)
    assert_array_almost_equal(conf_int, expected_conf_int)

    # now just the preds and stderr
    preds, stderr = arima.predict(n_periods=10, include_std_err=True)
    assert_array_almost_equal(preds, expected_preds)
    assert_array_almost_equal(stderr, expected_stderr)

    # now just the preds and confint
    preds, conf_int = arima.predict(n_periods=10, include_conf_int=True)
    assert_array_almost_equal(preds, expected_preds)
    assert_array_almost_equal(conf_int, expected_conf_int)

    # now just the preds
    preds = arima.predict(n_periods=10)
    assert_array_almost_equal(preds, expected_preds)


def test_more_elaborate():
    # show we can fit this with a non-zero order
    arima = ARIMA(order=(2, 1, 2)).fit(hr)

    # show we can get all these attrs without getting an error
    attrs = {
        'aic', 'arparams', 'arroots', 'bic', 'bse', 'df_model',
        'df_resid', 'hqic', 'k_ar', 'k_exog', 'k_ma', 'maparams',
        'maroots', 'params', 'pvalues', 'resid', 'sigma2', 'loglike'
    }

    # this just shows all of these attrs work.
    for attr in attrs:
        _ = getattr(arima, attr)()


def test_the_r_src():
    # this is the test the R code provides
    fit = ARIMA(order=(2, 0, 1)).fit(abc)

    # the R code's AIC = ~135
    assert abs(135 - fit.aic()) < 1.0

    # the R code's BIC = ~145
    assert abs(145 - fit.bic()) < 1.0

    # R's coefficients:
    #     ar1      ar2     ma1    mean
    # -0.6515  -0.2449  0.8012  5.0370

    # note that statsmodels' mean is on the front, not the end.
    params = fit.params()
    assert_almost_equal(params, np.array([5.0370, -0.6515, -0.2449, 0.8012]), decimal=2)

    # > fit = forecast::auto.arima(abc, max.p=5, max.d=5, max.q=5, max.order=100)
    fit = auto_arima(abc, max_p=5, max_d=5, max_q=5, max_order=100, suppress_warnings=True)

    # this differs from the R fit, but has a higher AIC, so the objective was achieved...
    assert abs(140 - fit.aic()) < 1.0


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
