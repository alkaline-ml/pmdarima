# -*- coding: utf-8 -*-
# stationarity tests

from pmdarima.arima.stationarity import ADFTest, PPTest, KPSSTest
from pmdarima.arima.utils import ndiffs
from pmdarima.utils.array import diff
from pmdarima.datasets import load_austres

from sklearn.utils import check_random_state
from numpy.testing import assert_array_almost_equal, assert_almost_equal, \
    assert_array_equal

import numpy as np
import pytest

# for testing rand of len 400 for m==365
random_state = check_random_state(42)
austres = load_austres()


def test_ndiffs_stationary():
    # show that for a stationary vector, ndiffs returns 0
    x = np.ones(10)
    assert ndiffs(x, alpha=0.05, test='kpss', max_d=2) == 0
    assert ndiffs(x, alpha=0.05, test='pp', max_d=2) == 0
    assert ndiffs(x, alpha=0.05, test='adf', max_d=2) == 0


@pytest.mark.parametrize("cls", (KPSSTest, PPTest, ADFTest))
def test_embedding(cls):
    x = np.arange(5)
    expected = np.array([
        [1, 2, 3, 4],
        [0, 1, 2, 3]
    ])

    assert_array_almost_equal(cls._embed(x, 2), expected)

    y = np.array([1, -1, 0, 2, -1, -2, 3])
    assert_array_almost_equal(cls._embed(y, 1),
                              np.array([
                                  [1, -1, 0, 2, -1, -2, 3]
                              ]))

    assert_array_almost_equal(cls._embed(y, 2).T,
                              np.array([
                                  [-1, 1],
                                  [0, -1],
                                  [2, 0],
                                  [-1, 2],
                                  [-2, -1],
                                  [3, -2]
                              ]))

    assert_array_almost_equal(cls._embed(y, 3).T,
                              np.array([
                                  [0, -1, 1],
                                  [2, 0, -1],
                                  [-1, 2, 0],
                                  [-2, -1, 2],
                                  [3, -2, -1]
                              ]))

    # Where K close to y dim
    assert_array_almost_equal(cls._embed(y, 6).T,
                              np.array([
                                  [-2, -1, 2, 0, -1, 1],
                                  [3, -2, -1, 2, 0, -1]
                              ]))

    # Where k == y dim
    assert_array_almost_equal(cls._embed(y, 7).T,
                              np.array([
                                  [3, -2, -1, 2, 0, -1, 1]
                              ]))

    # Assert we fail when k > dim
    with pytest.raises(ValueError):
        cls._embed(y, 8)


def test_adf_ols():
    # Test the _ols function of the ADF test
    x = np.array([1, -1, 0, 2, -1, -2, 3])
    k = 2
    y = diff(x)
    assert_array_equal(y, [-2, 1, 2, -3, -1, 5])

    z = ADFTest._embed(y, k).T
    res = ADFTest._ols(x, y, z, k)

    # Assert on the params of the OLS. The comparisons are those obtained
    # from the R function.
    expected = np.array([1.0522, -3.1825, -0.1609, 1.4690])
    assert np.allclose(res.params, expected, rtol=0.001)

    # Now assert on the standard error
    stat = ADFTest._ols_std_error(res)
    assert np.allclose(stat, -100.2895)  # derived from R code


def test_adf_p_value():
    # Assert on the ADF test's p-value
    p_val, do_diff = \
        ADFTest(alpha=0.05).should_diff(np.array([1, -1, 0, 2, -1, -2, 3]))

    assert np.isclose(p_val, 0.01)
    assert not do_diff


@pytest.mark.parametrize('null', ('level', 'trend'))
def test_kpss(null):
    test = KPSSTest(alpha=0.05, null=null, lshort=True)
    pval, do_diff = test.should_diff(austres)
    assert do_diff  # show it is significant
    assert_almost_equal(pval, 0.01)

    # Test on the data provided in issue #67
    x = np.array([1, -1, 0, 2, -1, -2, 3])
    pval2, do_diff2 = test.should_diff(x)

    # We expect Trend to be significant, but NOT Level
    if null == 'level':
        assert not do_diff2
        assert_almost_equal(pval2, 0.1)
    else:
        assert do_diff2
        assert_almost_equal(pval2, 0.01)

    # test the ndiffs with the KPSS test
    assert ndiffs(austres, test='kpss', max_d=5, null=null) == 2


def test_non_default_kpss():
    test = KPSSTest(alpha=0.05, null='trend', lshort=False)
    pval, do_diff = test.should_diff(austres)
    assert do_diff  # show it is significant
    assert np.allclose(pval, 0.01, atol=0.005)

    # test the ndiffs with the KPSS test
    assert ndiffs(austres, test='kpss', max_d=2) == 2


def test_kpss_corner():
    test = KPSSTest(alpha=0.05, null='something-else', lshort=True)
    with pytest.raises(ValueError):
        test.should_diff(austres)


def test_pp():
    test = PPTest(alpha=0.05, lshort=True)
    pval, do_diff = test.should_diff(austres)
    assert do_diff

    # Result from R code: 0.9786066
    # > pp.test(austres, lshort=TRUE)$p.value
    assert_almost_equal(pval, 0.9786066, decimal=5)

    # test n diffs
    assert ndiffs(austres, test='pp', max_d=2) == 1

    # If we use lshort is FALSE, it will be different
    test = PPTest(alpha=0.05, lshort=False)
    pval, do_diff = test.should_diff(austres)
    assert do_diff

    # Result from R code: 0.9514589
    # > pp.test(austres, lshort=FALSE)$p.value
    assert_almost_equal(pval, 0.9514589, decimal=5)
    assert ndiffs(austres, test='pp', max_d=2, lshort=False) == 1


def test_adf():
    # Test where k = 1
    test = ADFTest(alpha=0.05, k=1)
    pval, do_diff = test.should_diff(austres)

    # R's value: 0.8488036
    # > adf.test(austres, k=1, alternative='stationary')$p.value
    assert np.isclose(pval, 0.8488036)
    assert do_diff

    # Test for k = 2. R's value: 0.7060733
    # > adf.test(austres, k=2, alternative='stationary')$p.value
    test = ADFTest(alpha=0.05, k=2)
    pval, do_diff = test.should_diff(austres)
    assert np.isclose(pval, 0.7060733)
    assert do_diff

    # Test for k is None. R's value: 0.3493465
    # > adf.test(austres, alternative='stationary')$p.value
    test = ADFTest(alpha=0.05, k=None)
    pval, do_diff = test.should_diff(austres)
    assert np.isclose(pval, 0.3493465, rtol=0.0001)
    assert do_diff


def test_adf_corner():
    with pytest.raises(ValueError):
        ADFTest(alpha=0.05, k=-1)

    # show we can fit with k is None
    test = ADFTest(alpha=0.05, k=None)
    test.should_diff(austres)


def test_ndiffs_corner_cases():
    with pytest.raises(ValueError):
        ndiffs(austres, max_d=0)


def test_base_cases():
    classes = (ADFTest, KPSSTest, PPTest)
    for cls in classes:
        instance = cls()

        # Also show we get a warning with the deprecated func
        with pytest.warns(DeprecationWarning):
            p_val, is_stationary = instance.is_stationary(None)

        # results of base-case
        assert np.isnan(p_val)
        assert not is_stationary
