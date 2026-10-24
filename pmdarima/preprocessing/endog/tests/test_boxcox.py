# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats
import pytest
from decimal import Decimal, localcontext

from pmdarima.compat.pytest import pytest_error_str
from pmdarima.preprocessing import BoxCoxEndogTransformer

loggamma = stats.loggamma.rvs(5, size=500) + 5


@pytest.mark.parametrize(
    'X', [
        None,
        np.random.rand(loggamma.shape[0], 3),
    ]
)
def test_invertible(X):
    trans = BoxCoxEndogTransformer()
    y_t, e_t = trans.fit_transform(loggamma, X)
    y_prime, e_prime = trans.inverse_transform(y_t, X=e_t)

    assert_array_almost_equal(loggamma, y_prime)

    # X should all be the same too
    if X is None:
        assert X is e_t is e_prime is None
    else:
        assert_array_almost_equal(X, e_t)
        assert_array_almost_equal(X, e_prime)


def test_invertible_when_lambda_is_0():
    y = [1, 2, 3]
    trans = BoxCoxEndogTransformer(lmbda=0.)
    y_t, _ = trans.fit_transform(y)
    y_prime, _ = trans.inverse_transform(y_t)
    assert_array_almost_equal(y, y_prime)


@pytest.mark.parametrize('lmbda', [1e-20, -1e-20, 1e-8, -1e-8, 0., .5])
@pytest.mark.parametrize('shift', [0., 2.])
def test_boxcox_precision_near_zero(lmbda, shift):
    y = np.array([.25, 1., 4., 10.])
    original = y.copy()
    trans = BoxCoxEndogTransformer(lmbda=lmbda, lmbda2=shift)
    actual, _ = trans.fit_transform(y)
    with localcontext() as ctx:
        ctx.prec = 80
        lam = Decimal(str(lmbda))
        logs = [Decimal(str(value + shift)).ln() for value in y]
        expected = [float(((lam * log).exp() - 1) / lam)
                    if lam else float(log) for log in logs]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize('lmbda', [1e-20, -1e-20, 1e-8, -1e-8, 0., .5])
@pytest.mark.parametrize('shift', [0., 2.])
def test_inverse_boxcox_precision_near_zero(lmbda, shift):
    y = np.array([-.5, 0., .5, 1.])
    original = y.copy()
    trans = BoxCoxEndogTransformer(lmbda=lmbda, lmbda2=shift).fit([1., 2.])
    actual, _ = trans.inverse_transform(y)
    with localcontext() as ctx:
        ctx.prec = 80
        lam = Decimal(str(lmbda))
        values = [Decimal(str(value)) for value in y]
        expected = [float(((1 + lam * value).ln() / lam).exp()) - shift
                    if lam else float(value.exp()) - shift for value in values]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-14)
    np.testing.assert_array_equal(y, original)


def test_value_error_on_neg_lambda():
    trans = BoxCoxEndogTransformer(lmbda2=-4.)
    with pytest.raises(ValueError) as ve:
        trans.fit_transform([1, 2, 3])
    assert 'lmbda2 must be a non-negative' in pytest_error_str(ve)


class TestNonInvertibleBC:
    y = [-1., 0., 1.]

    def test_expected_error(self):
        y = self.y
        trans = BoxCoxEndogTransformer(lmbda=2.)
        with pytest.raises(ValueError):
            trans.fit_transform(y)

    def test_expected_warning(self):
        y = self.y
        trans = BoxCoxEndogTransformer(lmbda=2., neg_action="warn")
        with pytest.warns(UserWarning):
            y_t, _ = trans.fit_transform(y)

        # When we invert, it will not be the same
        y_prime, _ = trans.inverse_transform(y_t)
        assert not np.allclose(y_prime, y)

    def test_no_warning_on_ignore(self):
        y = self.y
        trans = BoxCoxEndogTransformer(lmbda=2., neg_action="ignore")
        y_t, _ = trans.fit_transform(y)

        # When we invert, it will not be the same
        y_prime, _ = trans.inverse_transform(y_t)
        assert not np.allclose(y_prime, y)

    def test_invertible_when_lam2(self):
        y = self.y
        trans = BoxCoxEndogTransformer(lmbda=2., lmbda2=2.)
        y_t, _ = trans.fit_transform(y)

        # When we invert, it will not be the same
        y_prime, _ = trans.inverse_transform(y_t)
        assert_array_almost_equal(y, y_prime)
