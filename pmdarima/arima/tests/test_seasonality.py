# -*- coding: utf-8 -*-
# seasonality tests

from __future__ import absolute_import

from pmdarima.arima.seasonality import CHTest
from pmdarima.arima.utils import nsdiffs

import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.utils.validation import check_random_state
import pytest

# TODO: redundant code with test_stationarity. Fix this in separate feature
austres = np.array([13067.3, 13130.5, 13198.4, 13254.2, 13303.7, 13353.9,
                    13409.3, 13459.2, 13504.5, 13552.6, 13614.3, 13669.5,
                    13722.6, 13772.1, 13832.0, 13862.6, 13893.0, 13926.8,
                    13968.9, 14004.7, 14033.1, 14066.0, 14110.1, 14155.6,
                    14192.2, 14231.7, 14281.5, 14330.3, 14359.3, 14396.6,
                    14430.8, 14478.4, 14515.7, 14554.9, 14602.5, 14646.4,
                    14695.4, 14746.6, 14807.4, 14874.4, 14923.3, 14988.7,
                    15054.1, 15121.7, 15184.2, 15239.3, 15288.9, 15346.2,
                    15393.5, 15439.0, 15483.5, 15531.5, 15579.4, 15628.5,
                    15677.3, 15736.7, 15788.3, 15839.7, 15900.6, 15961.5,
                    16018.3, 16076.9, 16139.0, 16203.0, 16263.3, 16327.9,
                    16398.9, 16478.3, 16538.2, 16621.6, 16697.0, 16777.2,
                    16833.1, 16891.6, 16956.8, 17026.3, 17085.4, 17106.9,
                    17169.4, 17239.4, 17292.0, 17354.2, 17414.2, 17447.3,
                    17482.6, 17526.0, 17568.7, 17627.1, 17661.5])

#  change the length to be longer so we can actually test the large case
aus_list = austres.tolist()  # type: list
austres_long = np.asarray(aus_list * 10)  # type: np.ndarray


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


@pytest.mark.parametrize('m', [
    # y len is now 1760, which is > 2 * m + 5
    365,

    # what if m > 365???
    366])
def test_ch_test_long(m):
    # TODO: what to assert??
    CHTest(m=m).estimate_seasonal_differencing_term(austres_long)


def test_ch_base():
    test = CHTest(m=2)
    assert test.estimate_seasonal_differencing_term(None) == 0

    # test really long m for random array
    random_state = check_random_state(42)
    CHTest(m=365).estimate_seasonal_differencing_term(random_state.rand(400))


def test_nsdiffs_corner_cases():
    # max_D must be a positive int
    with pytest.raises(ValueError):
        nsdiffs(austres, m=2, max_D=0)

    # assert 0 for constant
    assert nsdiffs([1, 1, 1, 1], m=2) == 0

    # show fails for m <= 1
    for m in (0, 1):
        with pytest.raises(ValueError):
            nsdiffs(austres, m=m)


def test_seas_dummy():
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
def test_sd_test(x, m, expected):
    res = CHTest._sd_test(x, m)
    assert np.allclose(res, expected)
