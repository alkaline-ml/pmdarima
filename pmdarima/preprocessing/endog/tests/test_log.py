# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import stats
import pytest

from pmdarima.preprocessing import LogEndogTransformer
from pmdarima.preprocessing import BoxCoxEndogTransformer

def test_same():
    y = [1, 2, 3]
    trans = BoxCoxEndogTransformer()
    log_trans = LogEndogTransformer()
    y_t, _ = trans.fit_transform(y)
    log_y_t, _ = log_trans.fit_transform(y)
    assert_array_almost_equal(log_y_t, y_t)
