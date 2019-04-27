# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pmdarima.compat._internal import get_pytest_mpl_threshold

import pmdarima as pm
import os
import pytest

# Some as numpy, some as series
datasets = [
    ['wineind', pm.datasets.load_wineind(True)],
    ['lynx', pm.datasets.load_lynx(False)],
    ['heartrate', pm.datasets.load_heartrate(True)],
    ['woolyrnq', pm.datasets.load_woolyrnq(True)],

    # Might need to add these datasets in, but not sure if
    # it's actually necessary...
    # ['austres', pm.datasets.load_austres(False)]
]

# We are ONLY going to run these tests if we are NOT on Travis,
# since Travis really doesn't play too nice with the different
# backends we have flying around out there.
travis = os.environ.get("TESTING_ON_TRAVIS", "false").lower() == "true"

if not travis:

    tolerance = get_pytest_mpl_threshold(
        {'Windows': 15, 'Darwin': 10, 'Linux': 10}
    )

    params = []
    for row in datasets:
        params.append(pytest.param(row[0], row[1]))

    @pytest.mark.parametrize('plot_type,dataset', params)
    @pytest.mark.mpl_image_compare(tolerance=tolerance)
    def test_plot_autocorrelations(plot_type, dataset):
        return pm.autocorr_plot(dataset, show=False).get_figure()

    @pytest.mark.parametrize('plot_type,dataset', params)
    @pytest.mark.mpl_image_compare(tolerance=tolerance)
    def test_plot_acf(plot_type, dataset):
        return pm.plot_acf(dataset, show=False)

    @pytest.mark.parametrize('plot_type,dataset', params)
    @pytest.mark.mpl_image_compare(tolerance=tolerance)
    def test_plot_pacf(plot_type, dataset):
        return pm.plot_pacf(dataset, show=False)
