# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pyramid as pm
from pyramid.utils.wrapped import acf, pacf

import statsmodels.api as sm
import numpy as np

y = pm.datasets.load_wineind()


def test_wrapped_functions():
    for wrapped, native in ((sm.tsa.stattools.acf, acf),
                            (sm.tsa.stattools.pacf, pacf)):

        sm_res = wrapped(y)  # type: np.ndarray
        pm_res = native(y)
        assert np.allclose(sm_res, pm_res)

        # Show the docstrings are the same
        assert wrapped.__doc__ == native.__doc__
