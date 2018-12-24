# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pmdarima.datasets import load_lynx
from pmdarima.arima import ARIMA

import pytest
import os
import platform

lynx = load_lynx()

# test images directories
travis = os.environ.get("TESTING_ON_TRAVIS", "false").lower() == "true"

# Do not test on travis because they hate MPL
if not travis:

    # base images are created on Mac/Darwin. Windows needs a higher tolerance
    if platform.system() == "Windows":
        tolerance = 10
    else:
        tolerance = 5

    @pytest.mark.parametrize(
        'model_type,model', [
            pytest.param('arma', ARIMA(order=(1, 0, 0))),
            pytest.param('arima', ARIMA(order=(1, 1, 0))),
            pytest.param('sarimax', ARIMA(order=(1, 1, 0),
                                          seasonal_order=(1, 0, 0, 12)))
        ])
    @pytest.mark.mpl_image_compare(tolerance=tolerance)
    def test_plot_diagnostics(model_type, model):
        model.fit(lynx)
        return model.plot_diagnostics(figsize=(15, 12))
