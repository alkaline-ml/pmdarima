# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pyramid.arima._arima import C_is_not_finite

import numpy as np


def test_not_finite():
    assert C_is_not_finite(np.nan)
    assert C_is_not_finite(np.inf)
    assert not C_is_not_finite(5.)
