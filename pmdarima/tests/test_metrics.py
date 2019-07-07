# -*- coding: utf-8 -*-

from pmdarima.metrics import smape
import numpy as np
import pytest


@pytest.mark.parametrize(
    'actual,forecasted,expected', [
        pytest.param([0.07533, 0.07533, 0.07533, 0.07533,
                      0.07533, 0.07533, 0.0672, 0.0672],
                     [0.102, 0.107, 0.047, 0.1,
                      0.032, 0.047, 0.108, 0.089], 42.60306631890196),

        # when y_true == y_pred, we get 0 err
        pytest.param([0.07533, 0.07533, 0.07533, 0.07533,
                      0.07533, 0.07533, 0.0672, 0.0672],
                     [0.07533, 0.07533, 0.07533, 0.07533,
                      0.07533, 0.07533, 0.0672, 0.0672], 0),
    ]
)
def test_smape(actual, forecasted, expected):
    err = smape(actual, forecasted)
    assert np.allclose(expected, err)
