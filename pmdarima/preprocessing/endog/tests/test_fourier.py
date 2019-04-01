# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import fftpack
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
