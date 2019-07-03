# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pmdarima.datasets import load_lynx
from pmdarima.arima import ARIMA
from pmdarima.compat._internal import get_pytest_mpl_threshold

import pytest
import os

lynx = load_lynx()

# test images directories
travis = os.environ.get("TESTING_ON_TRAVIS", "false").lower() == "true"

# TODO: should we remove this test suite altogether? It seems a bit brittle

# Do not test on travis because they hate MPL
if not travis:

    # Give tolerance of 10 to all for now. In the past, Windows needed more.
    tolerance = get_pytest_mpl_threshold(
        {'Windows': 10, 'Darwin': 10, 'Linux': 10}
    )

    @pytest.mark.parametrize(
        'model_type,model', [
            pytest.param('arma', ARIMA(order=(1, 0, 0), maxiter=50)),
            pytest.param('arima', ARIMA(order=(1, 1, 0), maxiter=50)),
            pytest.param('sarimax', ARIMA(order=(1, 1, 0),
                                          maxiter=50,
                                          seasonal_order=(1, 0, 0, 12)))
        ])
    @pytest.mark.mpl_image_compare(tolerance=tolerance)
    def test_plot_diagnostics(model_type, model):
        model.fit(lynx)
        return model.plot_diagnostics(figsize=(15, 12))
