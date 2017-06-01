
from __future__ import absolute_import
from pyramid.arima import ARIMA, auto_arima
from pyramid.arima.auto import _fmt_warning_str
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

wineind = np.array([15136, 16733, 20016, 17708, 18019, 19227, 22893, 23739, 21133,
                    22591, 26786, 29740, 15028, 17977, 20008, 21354, 19498, 22125,
                    25817, 28779, 20960, 22254, 27392, 29945, 16933, 17892, 20533,
                    23569, 22417, 22084, 26580, 27454, 24081, 23451, 28991, 31386,
                    16896, 20045, 23471, 21747, 25621, 23859, 25500, 30998, 24475,
                    23145, 29701, 34365, 17556, 22077, 25702, 22214, 26886, 23191,
                    27831, 35406, 23195, 25110, 30009, 36242, 18450, 21845, 26488,
                    22394, 28057, 25451, 24872, 33424, 24052, 28449, 33533, 37351,
                    19969, 21701, 26249, 24493, 24603, 26485, 30723, 34569, 26689,
                    26157, 32064, 38870, 21337, 19419, 23166, 28286, 24570, 24001,
                    33151, 24878, 26804, 28967, 33311, 40226, 20504, 23060, 23562,
                    27562, 23940, 24584, 34303, 25517, 23494, 29095, 32903, 34379,
                    16991, 21109, 23740, 25552, 21752, 20294, 29009, 25500, 24166,
                    26960, 31222, 38641, 14672, 17543, 25453, 32683, 22449, 22316,
                    27595, 25451, 25421, 25288, 32568, 35110, 16052, 22146, 21198,
                    19543, 22084, 23816, 29961, 26773, 26635, 26972, 30207, 38687,
                    16974, 21697, 24179, 23757, 25013, 24019, 30345, 24488, 25156,
                    25650, 30923, 37240, 17466, 19463, 24352, 26805, 25236, 24735,
                    29356, 31234, 22724, 28496, 32857, 37198, 13652, 22784, 23565,
                    26323, 23779, 27549, 29660, 23356])

lynx = np.array([269,  321,  585,  871,  1475, 2821, 3928, 5943, 4950, 2577, 523,  98,   184,  279,  409,
                 2285, 2685, 3409, 1824, 409,  151,  45,   68,   213,  546,  1033, 2129, 2536, 957,  361,
                 377,  225,  360,  731,  1638, 2725, 2871, 2119, 684,  299,  236,  245,  552,  1623, 3311,
                 6721, 4254, 687,  255,  473,  358,  784,  1594, 1676, 2251, 1426, 756,  299,  201,  229,
                 469,  736,  2042, 2811, 4431, 2511, 389,  73,   39,   49,   59,   188,  377,  1292, 4031,
                 3495, 587,  105,  153,  387,  758,  1307, 3465, 6991, 6313, 3794, 1836, 345,  382,  808,
                 1388, 2713, 3800, 3091, 2985, 3790, 674,  81,   80,   108,  229,  399,  1132, 2432, 3574,
                 2935, 1537, 529,  485,  662,  1000, 1590, 2657, 3396])


def test_basic_arima():
    arima = ARIMA(order=(0, 0, 0), trend='c').fit(y)

    # test some of the attrs
    assert_almost_equal(arima.aic(), 11.201308403566909, decimal=5)
    assert_almost_equal(arima.bic(), 13.639060053303311, decimal=5)

    # get predictions
    expected_preds = np.array([0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876, 0.44079876, 0.44079876,
                               0.44079876])

    # generate predictions
    preds = arima.predict(n_periods=10)
    assert_array_almost_equal(preds, expected_preds)


def _try_get_attrs(arima):
    # show we can get all these attrs without getting an error
    attrs = {
        'aic', 'arparams', 'arroots', 'bic', 'bse',
        'df_resid', 'hqic', 'maparams', 'maroots',
        'params', 'pvalues', 'resid',
    }

    # this just shows all of these attrs work.
    for attr in attrs:
        _ = getattr(arima, attr)()


def test_more_elaborate():
    # show we can fit this with a non-zero order
    arima = ARIMA(order=(2, 1, 2)).fit(y=hr)
    _try_get_attrs(arima)


def test_the_r_src():
    # this is the test the R code provides
    fit = ARIMA(order=(2, 0, 1), trend='c').fit(abc)

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
    fit = auto_arima(abc, max_p=5, max_d=5, max_q=5, max_order=100, seasonal=False,
                     trend='c', suppress_warnings=True, error_action='ignore')

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

    # show bad m value
    _assert_val_error(auto_arima, abc, m=0)


def test_many_orders():
    # show that auto-arima can't fit this data for some reason...
    lam = 0.5
    lynx_bc = ((lynx ** lam) - 1) / lam

    failed = False
    try:
        auto_arima(lynx_bc, start_p=1, start_q=1, d=0, max_p=5, max_q=5, n_jobs=-1)
    except ValueError:
        failed = True
    assert failed


def test_with_seasonality():
    fit = ARIMA(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(y=wineind)
    _try_get_attrs(fit)

    # R code AIC result is ~3004
    assert abs(fit.aic() - 3004) < 100  # show equal within 100 or so

    # R code BIC result is ~3017
    assert abs(fit.bic() - 3017) < 100  # show equal within 100 or so

    # can we auto-arima this?
    seasonal_fit = auto_arima(wineind, start_p=1, start_q=1, max_p=2, max_q=2, m=12,
                              start_P=0, seasonal=True, n_jobs=-1, d=1, D=1,
                              error_action='raise')  # do raise so it fails fast

    # show that we can forecast even after the pickling (this was fit in parallel)
    seasonal_fit.predict(n_periods=10)


def test_warning_str_fmt():
    order = (1, 1, 1)
    seasonal = (1, 1, 1, 1)
    for ssnl in (seasonal, None):
        _ = _fmt_warning_str(order, ssnl)
