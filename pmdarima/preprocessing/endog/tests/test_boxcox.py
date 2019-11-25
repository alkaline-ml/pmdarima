# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats
import pytest

from pmdarima.compat.pytest import pytest_error_str
from pmdarima.preprocessing import BoxCoxEndogTransformer

loggamma = stats.loggamma.rvs(5, size=500) + 5


@pytest.mark.parametrize(
    'exog', [
        None,
        np.random.rand(loggamma.shape[0], 3),
    ]
)
def test_invertible(exog):
    trans = BoxCoxEndogTransformer()
    y_t, e_t = trans.fit_transform(loggamma, exogenous=exog)
    y_prime, e_prime = trans.inverse_transform(y_t, exogenous=e_t)

    assert_array_almost_equal(loggamma, y_prime)

    # exog should all be the same too
    if exog is None:
        assert exog is e_t is e_prime is None
    else:
        assert_array_almost_equal(exog, e_t)
        assert_array_almost_equal(exog, e_prime)


def test_invertible_when_lambda_is_0():
    y = [1, 2, 3]
    trans = BoxCoxEndogTransformer(lmbda=0.)
    y_t, _ = trans.fit_transform(y)
    y_prime, _ = trans.inverse_transform(y_t)
    assert_array_almost_equal(y, y_prime)


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
