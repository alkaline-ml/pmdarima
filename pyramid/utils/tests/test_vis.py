# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pyramid as pm
from matplotlib.testing.decorators import cleanup

# Some as numpy, some as series
datasets = [
    pm.datasets.load_wineind(True),
    pm.datasets.load_lynx(False),
    pm.datasets.load_heartrate(True)
]


@cleanup
def do_plot(plotting_func, dataset):
    plotting_func(dataset, block=False)


def test_plot_autocorrelations():
    for ds in datasets:
        do_plot(pm.autocorr_plot, ds)


def test_plot_acf():
    for ds in datasets:
        do_plot(pm.plot_acf, ds)


def test_plot_pacf():
    for ds in datasets:
        do_plot(pm.plot_pacf, ds)
