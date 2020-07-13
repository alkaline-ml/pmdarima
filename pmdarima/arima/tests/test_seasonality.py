# -*- coding: utf-8 -*-
# seasonality tests

from pmdarima.arima.seasonality import CHTest, decompose, OCSBTest
from pmdarima.arima.utils import nsdiffs
from pmdarima.compat.pytest import pytest_error_str
from pmdarima.datasets import \
    load_airpassengers, load_ausbeer, load_austres, load_wineind

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils.validation import check_random_state
import pytest

from unittest import mock

airpassengers = load_airpassengers()
austres = load_austres()
ausbeer = load_ausbeer()
wineind = load_wineind()

#  change the length to be longer so we can actually test the large case
aus_list = austres.tolist()  # type: list
austres_long = np.asarray(aus_list * 10)  # type: np.ndarray


@pytest.mark.parametrize(
    'x,type_,m,filter_', [
        pytest.param(ausbeer, 'additive', 4, None),
        pytest.param(airpassengers, 'multiplicative', 12, None),
        pytest.param(wineind, 'additive', 12, None),
        pytest.param(np.array([1., 2., 3., 4., 5., 6.]), 'additive', 3, None)
    ]
)
def test_decompose_happy_path(x, type_, m, filter_):

    decomposed_tuple = decompose(x, type_, m, filter_)
    first_ind = int(m / 2)
    last_ind = -int(m / 2)
    x = decomposed_tuple.x[first_ind:last_ind]
    trend = decomposed_tuple.trend[first_ind:last_ind]
    seasonal = decomposed_tuple.seasonal[first_ind:last_ind]
    random = decomposed_tuple.random[first_ind:last_ind]

    if type_ == 'multiplicative':
        reconstructed_x = trend * seasonal * random
    else:
        reconstructed_x = trend + seasonal + random

    assert_almost_equal(x, reconstructed_x)


def test_decompose_corner_cases():
    with pytest.raises(ValueError):
        decompose(ausbeer, 'dummy_type', 4, None),  # bad `type_`

    with pytest.raises(ValueError):
        decompose(airpassengers, 'multiplicative', -0.5, None),  # bad `m`

    with pytest.raises(ValueError):
        decompose(ausbeer[:1], 'multiplicative', 4, None)  # bad `x`


@pytest.mark.parametrize(
    'm,expected', [
        pytest.param(3, 0),
        pytest.param(24, 0),
        pytest.param(52, 0),
        pytest.param(365, 0)
    ]
)
def test_ch_test_m_values(m, expected):
    assert CHTest(m=m).estimate_seasonal_differencing_term(austres) == expected


@pytest.mark.parametrize(
    'm,chstat,expected', [
        pytest.param(365, 66., 1),
        pytest.param(365, 63., 0),
        pytest.param(366, 65., 1),
        pytest.param(366, 60., 0),
    ]
)
def test_ch_test_long(m, chstat, expected):
    chtest = CHTest(m=m)
    y = np.random.rand(m * 3)  # very long, but mock makes it not matter

    mock_sdtest = (lambda *args, **kwargs: chstat)
    with mock.patch.object(chtest, '_sd_test', mock_sdtest):
        res = chtest.estimate_seasonal_differencing_term(y)

    assert expected == res


def test_ch_base():
    test = CHTest(m=2)
    assert test.estimate_seasonal_differencing_term(None) == 0

    # test really long m for random array
    random_state = check_random_state(42)
    CHTest(m=365).estimate_seasonal_differencing_term(random_state.rand(400))


@pytest.mark.parametrize(
    'tst', ('ocsb', 'ch')
)
def test_nsdiffs_corner_cases(tst):
    # max_D must be a positive int
    with pytest.raises(ValueError):
        nsdiffs(austres, m=2, max_D=0, test=tst)

    # assert 0 for constant
    assert nsdiffs([1, 1, 1, 1], m=2, test=tst) == 0

    # show fails for m <= 1
    for m in (0, 1):
        with pytest.raises(ValueError):
            nsdiffs(austres, m=m, test=tst)


def test_ch_seas_dummy():
    x = austres

    # Results from R. Don't try this in the console; it tends to
    # freak out and fall apart...
    expected = np.array([
        [6.123234e-17, 1.000000e+00, -1],
        [-1.000000e+00, 1.224647e-16, 1],
        [-1.836970e-16, -1.000000e+00, -1],
        [1.000000e+00, -2.449294e-16, 1],
        [3.061617e-16, 1.000000e+00, -1],
        [-1.000000e+00, 3.673940e-16, 1],
        [-4.286264e-16, -1.000000e+00, -1],
        [1.000000e+00, -4.898587e-16, 1],
        [5.510911e-16, 1.000000e+00, -1],
        [-1.000000e+00, 6.123234e-16, 1],
        [-2.449913e-15, -1.000000e+00, -1],
        [1.000000e+00, -7.347881e-16, 1],
        [-9.803364e-16, 1.000000e+00, -1],
        [-1.000000e+00, 8.572528e-16, 1],
        [-2.694842e-15, -1.000000e+00, -1],
        [1.000000e+00, -9.797174e-16, 1],
        [-7.354071e-16, 1.000000e+00, -1],
        [-1.000000e+00, 1.102182e-15, 1],
        [-2.939771e-15, -1.000000e+00, -1],
        [1.000000e+00, -1.224647e-15, 1],
        [-4.904777e-16, 1.000000e+00, -1],
        [-1.000000e+00, 4.899825e-15, 1],
        [-3.184701e-15, -1.000000e+00, -1],
        [1.000000e+00, -1.469576e-15, 1],
        [-2.455483e-16, 1.000000e+00, -1],
        [-1.000000e+00, -1.960673e-15, 1],
        [-3.429630e-15, -1.000000e+00, -1],
        [1.000000e+00, -1.714506e-15, 1],
        [-6.189806e-19, 1.000000e+00, -1],
        [-1.000000e+00, 5.389684e-15, 1],
        [-3.674559e-15, -1.000000e+00, -1],
        [1.000000e+00, -1.959435e-15, 1],
        [2.443104e-16, 1.000000e+00, -1],
        [-1.000000e+00, -1.470814e-15, 1],
        [-3.919489e-15, -1.000000e+00, -1],
        [1.000000e+00, -2.204364e-15, 1],
        [4.892397e-16, 1.000000e+00, -1],
        [-1.000000e+00, 5.879543e-15, 1],
        [-4.164418e-15, -1.000000e+00, -1],
        [1.000000e+00, -2.449294e-15, 1],
        [7.839596e-15, 1.000000e+00, -1],
        [-1.000000e+00, -9.809554e-16, 1],
        [-4.409347e-15, -1.000000e+00, -1],
        [1.000000e+00, -9.799650e-15, 1],
        [9.790985e-16, 1.000000e+00, -1],
        [-1.000000e+00, 6.369401e-15, 1],
        [2.451151e-15, -1.000000e+00, -1],
        [1.000000e+00, -2.939152e-15, 1],
        [8.329455e-15, 1.000000e+00, -1],
        [-1.000000e+00, -4.910967e-16, 1],
        [-4.899206e-15, -1.000000e+00, -1],
        [1.000000e+00, 3.921346e-15, 1],
        [1.468957e-15, 1.000000e+00, -1],
        [-1.000000e+00, 6.859260e-15, 1],
        [1.961292e-15, -1.000000e+00, -1],
        [1.000000e+00, -3.429011e-15, 1],
        [8.819314e-15, 1.000000e+00, -1],
        [-1.000000e+00, -1.237961e-18, 1],
        [-5.389065e-15, -1.000000e+00, -1],
        [1.000000e+00, -1.077937e-14, 1],
        [1.958816e-15, 1.000000e+00, -1],
        [-1.000000e+00, 7.349119e-15, 1],
        [1.471433e-15, -1.000000e+00, -1],
        [1.000000e+00, -3.918870e-15, 1],
        [9.309173e-15, 1.000000e+00, -1],
        [-1.000000e+00, 4.886208e-16, 1],
        [-5.878924e-15, -1.000000e+00, -1],
        [1.000000e+00, 2.941628e-15, 1],
        [2.448675e-15, 1.000000e+00, -1],
        [-1.000000e+00, 7.838977e-15, 1],
        [9.815744e-16, -1.000000e+00, -1],
        [1.000000e+00, -4.408728e-15, 1],
        [9.799031e-15, 1.000000e+00, -1],
        [-1.000000e+00, 9.784795e-16, 1],
        [-6.368782e-15, -1.000000e+00, -1],
        [1.000000e+00, -1.175909e-14, 1],
        [2.938533e-15, 1.000000e+00, -1],
        [-1.000000e+00, 8.328836e-15, 1],
        [4.917157e-16, -1.000000e+00, -1],
        [1.000000e+00, -4.898587e-15, 1],
        [1.028889e-14, 1.000000e+00, -1],
        [-1.000000e+00, 1.567919e-14, 1],
        [7.352214e-15, -1.000000e+00, -1],
        [1.000000e+00, 1.961911e-15, 1],
        [3.428392e-15, 1.000000e+00, -1],
        [-1.000000e+00, 8.818695e-15, 1],
        [-1.420900e-14, -1.000000e+00, -1],
        [1.000000e+00, -1.959930e-14, 1],
        [-3.432106e-15, 1.000000e+00, -1]
    ])

    actual = CHTest._seas_dummy(x, 4)
    assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    'x,m,expected', [
        pytest.param(austres, 3, 0.07956102),  # R code produces 0.07956102
        pytest.param(austres, 4, 0.1935046),  # Expected from R: 0.1935046
        pytest.param(austres, 24, 4.134289)  # R res: 4.134289
    ]
)
def test_ch_sd_test(x, m, expected):
    res = CHTest._sd_test(x, m)
    assert np.allclose(res, expected)


def test_ocsb_do_lag():
    q = np.arange(5)

    assert_array_equal(OCSBTest._do_lag(q, 1, False),
                       [[0.],
                        [1.],
                        [2.],
                        [3.],
                        [4.]])

    assert_array_equal(OCSBTest._do_lag(q, 1, True),
                       [[0.],
                        [1.],
                        [2.],
                        [3.],
                        [4.]])

    assert_array_equal(OCSBTest._do_lag(q, 2, False),
                       [[0., np.nan],
                        [1., 0.],
                        [2., 1.],
                        [3., 2.],
                        [4., 3.],
                        [np.nan, 4.]])

    assert_array_equal(OCSBTest._do_lag(q, 2, True),
                       [[1., 0.],
                        [2., 1.],
                        [3., 2.],
                        [4., 3.]])

    assert_array_equal(OCSBTest._do_lag(q, 3, False),
                       [[0., np.nan, np.nan],
                        [1., 0., np.nan],
                        [2., 1., 0.],
                        [3., 2., 1.],
                        [4., 3., 2.],
                        [np.nan, 4., 3.],
                        [np.nan, np.nan, 4.]])

    assert_array_equal(OCSBTest._do_lag(q, 3, True),
                       [[2., 1., 0.],
                        [3., 2., 1.],
                        [4., 3., 2.]])

    assert_array_equal(OCSBTest._do_lag(q, 4, False),
                       [[0., np.nan, np.nan, np.nan],
                        [1., 0., np.nan, np.nan],
                        [2., 1., 0., np.nan],
                        [3., 2., 1., 0.],
                        [4., 3., 2., 1.],
                        [np.nan, 4., 3., 2.],
                        [np.nan, np.nan, 4., 3.],
                        [np.nan, np.nan, np.nan, 4.]])

    assert_array_equal(OCSBTest._do_lag(q, 4, True),
                       [[3., 2., 1., 0.],
                        [4., 3., 2., 1.]])


def test_ocsb_gen_lags():
    z_res = OCSBTest._gen_lags(austres, 0)
    assert z_res.shape == austres.shape
    assert (z_res == 0).all()


@pytest.mark.parametrize(
    'lag_method,expected,max_lag', [
        # ocsb.test(austres, lag.method='fixed', maxlag=2)$stat -> -5.673749
        pytest.param('fixed', -5.6737, 2),

        # ocsb.test(austres, lag.method='fixed', maxlag=3)$stat -> -5.632227
        pytest.param('fixed', -5.6280, 3),

        # ocsb.test(austres, lag.method='AIC', maxlag=2)$stat -> -6.834392
        # We get a singular matrix error in Python that doesn't show up in R,
        # but we found a way to recover. Unforunately, it means our results are
        # different...
        pytest.param('aic', -5.66870, 2),
        pytest.param('aic', -6.03761, 3),
        pytest.param('bic', -5.66870, 2),
        pytest.param('bic', -6.03761, 3),
        pytest.param('aicc', -5.66870, 2),
        pytest.param('aicc', -6.03761, 3),
    ]
)
def test_ocsb_test_statistic(lag_method, expected, max_lag):
    test = OCSBTest(m=4, max_lag=max_lag, lag_method=lag_method)
    test_stat = test._compute_test_statistic(austres)
    assert np.allclose(test_stat, expected, rtol=0.01)


def test_ocsb_regression():
    # fitOCSB is a closure function inside of forecast::ocsb.test
    # > fitOCSB(austres, 1, 1)
    # Coefficients:
    # xregmf.x    xregZ4    xregZ5
    #   0.2169    0.2111   -0.8625

    # We get different results here, but only marginally...
    reg = OCSBTest._fit_ocsb(austres, m=4, lag=1, max_lag=1)
    coef = reg.params
    assert np.allclose(coef, [0.2169, 0.2111, -0.8625], rtol=0.01)


def test_failing_ocsb():
    # TODO: should this pass?
    # This passes in R, but statsmodels can't compute the regression...
    with pytest.raises(ValueError):
        OCSBTest(m=4, max_lag=0).estimate_seasonal_differencing_term(austres)

    # Fail for bad method
    with pytest.raises(ValueError) as v:
        OCSBTest(m=4, max_lag=3, lag_method="bad_method")\
            .estimate_seasonal_differencing_term(austres)
    assert "invalid method" in pytest_error_str(v)
