# -*- coding: utf-8 -*-

from pmdarima.compat import sklearn as sk
import numpy as np
from numpy.testing import assert_array_equal
import pytest


@pytest.mark.parametrize(
    'x,i,exp', [
        pytest.param(np.array([1, 2, 3, 4, 5]), [0, 1], np.array([1, 2])),
        pytest.param(np.array([[1, 2], [3, 4]]), [0], np.array([[1, 2]])),
    ]
)
def test_safe_indexing(x, i, exp):
    res = sk.safe_indexing(x, i)
    assert_array_equal(exp, res)
