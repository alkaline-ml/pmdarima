# stationarity/seasonality tests

from __future__ import absolute_import

from pyramid.arima.stationarity import ADFTest, PPTest, KPSSTest
from pyramid.arima.seasonality import CHTest
from pyramid.arima.utils import ndiffs, nsdiffs

from sklearn.utils import check_random_state
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import numpy as np
import pytest

# for testing rand of len 400 for m==365
random_state = check_random_state(42)

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


def test_ndiffs_stationary():
    # show that for a stationary vector, ndiffs returns 0
    x = np.ones(10)
    assert ndiffs(x, alpha=0.05, test='kpss', max_d=2) == 0
    assert ndiffs(x, alpha=0.05, test='pp', max_d=2) == 0
    assert ndiffs(x, alpha=0.05, test='adf', max_d=2) == 0


def test_embedding():
    x = np.arange(5)
    expected = np.array([
        [1, 2, 3, 4],
        [0, 1, 2, 3]
    ])

    assert_array_almost_equal(KPSSTest()._embed(x, 2), expected)


def test_kpss():
    test = KPSSTest(alpha=0.05, null='level', lshort=True)
    pval, is_sig = test.is_stationary(austres)
    assert is_sig  # show it is significant
    assert_almost_equal(pval, 0.01)

    # test the ndiffs with the KPSS test
    assert ndiffs(austres, test='kpss', max_d=2) == 2


def test_non_default_kpss():
    test = KPSSTest(alpha=0.05, null='trend', lshort=False)
    pval, is_sig = test.is_stationary(austres)
    assert is_sig  # show it is significant
    assert_almost_equal(pval, 0.01)

    # test the ndiffs with the KPSS test
    assert ndiffs(austres, test='kpss', max_d=2) == 2


def test_kpss_corner():
    test = KPSSTest(alpha=0.05, null='something-else', lshort=True)
    with pytest.raises(ValueError):
        test.is_stationary(austres)


def test_pp():
    test = PPTest(alpha=0.05, lshort=True)
    pval, is_sig = test.is_stationary(austres)
    assert is_sig
    assert_almost_equal(pval, 0.02139, decimal=5)

    # test n diffs
    nd = ndiffs(austres, test='pp', max_d=2)
    assert nd == 1


def test_adf():
    test = ADFTest(alpha=0.05, k=2)
    pval, is_sig = test.is_stationary(austres)
    assert not is_sig

    # OLS in statsmodels seems unstable compared to the R code?...


def test_adf_corner():
    with pytest.raises(ValueError):
        ADFTest(alpha=0.05, k=-1)

    # show we can fit with k is None
    test = ADFTest(alpha=0.05, k=None)
    test.is_stationary(austres)


def test_ch_test():
    val = CHTest._sd_test(austres, 3)

    # R code produces 0.07956102
    assert_almost_equal(val, 0.07956102, decimal=7)
    assert CHTest(m=3).estimate_seasonal_differencing_term(austres) == 0

    # what if freq > 12?
    assert_almost_equal(CHTest._sd_test(austres, 24), 4.134289, decimal=5)
    assert CHTest(m=24).estimate_seasonal_differencing_term(austres) == 0
    assert CHTest(m=52).estimate_seasonal_differencing_term(austres) == 0

    # this won't even go thru because n < 2 * m + 5:
    assert CHTest(m=365).estimate_seasonal_differencing_term(austres) == 0

    # change the length to be longer so we can actually test the end case
    aus_list = austres.tolist()  # type: list
    y = np.asarray(aus_list * 10)  # type: np.ndarray

    # y len is now 1760, which is > 2 * m + 5, but idk what to assert
    CHTest(m=365).estimate_seasonal_differencing_term(y)

    # what if m > 365???
    CHTest(m=366).estimate_seasonal_differencing_term(y)


def test_ch_base():
    test = CHTest(m=2)
    assert test.estimate_seasonal_differencing_term(None) == 0

    # test really long m for random array
    CHTest(m=365).estimate_seasonal_differencing_term(random_state.rand(400))


def test_ndiffs_corner_cases():
    with pytest.raises(ValueError):
        ndiffs(austres, max_d=0)


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


def test_base_cases():
    classes = (ADFTest, KPSSTest, PPTest)
    for cls in classes:
        instance = cls()
        p_val, is_stationary = instance.is_stationary(None)

        # results of base-case
        assert np.isnan(p_val)
        assert not is_stationary
