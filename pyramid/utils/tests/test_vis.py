# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pyramid as pm
import os

# Some as numpy, some as series
datasets = [
    pm.datasets.load_wineind(True),
    pm.datasets.load_lynx(False),
    pm.datasets.load_heartrate(True),
    pm.datasets.load_woolyrnq(True)
]

# We are ONLY going to run these tests if we are NOT on Travis,
# since Travis really doesn't play too nice with the different
# backends we have flying around out there.
travis = os.environ.get("TESTING_ON_TRAVIS", "false").lower() == "true"


def do_plot(plotting_func, dataset):
    if not travis:
        plotting_func(dataset, show=False)


def test_plot_autocorrelations():
    for ds in datasets:
        do_plot(pm.autocorr_plot, ds)


def test_plot_acf():
    for ds in datasets:
        do_plot(pm.plot_acf, ds)


def test_plot_pacf():
    for ds in datasets:
        do_plot(pm.plot_pacf, ds)
