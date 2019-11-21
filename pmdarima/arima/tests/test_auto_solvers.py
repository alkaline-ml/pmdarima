# -*- coding: utf-8 -*-

from pmdarima.arima._auto_solvers import _do_root_test
from numpy.testing import assert_almost_equal
import numpy as np
import pytest


@pytest.mark.parametrize(
    'coef,sign,expected', [
        pytest.param([1., -0.080619838, -0.442994620], 1, 1.41421138),
    ]
)
def test_do_root_test(coef, sign, expected):
    minroot = _do_root_test(np.array(coef), 2, sign)
    assert_almost_equal(minroot, expected)
