# -*- coding: utf-8 -*-

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from pmdarima.compat import pmdarima as pm_compat
from pmdarima.compat import pytest as pt_compat

xreg = np.random.rand(4, 4)


@pytest.mark.parametrize(
    'X,kw,X_exp,kw_exp,exp_warning,exp_error', [

        # provided as `exogenous`
        pytest.param(
            None,
            {"exogenous": xreg},
            xreg,
            {},
            DeprecationWarning,
            None,
        ),

        # provided as `X` with additional kwargs
        pytest.param(
            xreg,
            {"foo": "bar"},
            xreg,
            {"foo": "bar"},
            None,
            None,
        ),

        # provided as `X` AND `exogenous` will raise
        pytest.param(
            xreg,
            {"exogenous": xreg},
            None,
            None,
            None,
            ValueError,
        ),

    ]
)
def test_get_X(X, kw, X_exp, kw_exp, exp_warning, exp_error):
    with pytest.warns(exp_warning) as w, \
            pt_compat.raises(exp_error) as e:
        X_act, kw_act = pm_compat.get_X(X, **kw)

    if exp_warning:
        assert w
    else:
        assert not w

    if exp_error:
        assert e
        # no other assertions can be made
    else:
        assert not e
        assert_array_equal(X_act, X_exp)
        assert kw_act == kw_exp
