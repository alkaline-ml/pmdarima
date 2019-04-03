# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pmdarima.preprocessing import FourierEndogTransformer


@pytest.mark.parametrize(
    'exog', [
        None,
        np.random.rand(5, 3),
    ]
)
def test_invertible(exog):
    y = np.arange(5)
    trans = FourierEndogTransformer()
    y_t, e_t = trans.fit_transform(y, exog)
    y_prime, e_prime = trans.inverse_transform(y_t, e_t)

    assert_array_almost_equal(y, y_prime)

    # exog should all be the same too
    if exog is None:
        assert exog is e_t is e_prime is None
    else:
        assert_array_almost_equal(exog, e_t)
        assert_array_almost_equal(exog, e_prime)


def test_value_error_when_y_is_none():
    y = None
    exog = None
    trans = FourierEndogTransformer()
    with pytest.raises(ValueError):
        trans.fit_transform(y, exog)


def test_type_error_when_y_is_not_real():
    y = [4. + 0.j, 8. + 12.j, 16. + 0.j, 8. - 12.j]
    exog = None
    trans = FourierEndogTransformer()
    with pytest.raises(TypeError):
        trans.fit_transform(y, exog)


def test_hyndman_blog():
    n = 2000
    m = 200
    y = np.random.RandomState(1).normal(size=n) + (np.arange(1, n + 1) % 100 / 30)
