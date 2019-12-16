# -*- coding: utf-8 -*-

from unittest.mock import patch
import numpy as np
import pytest


class MockACPlot:
    def __init__(self, series):
        self.series = series
        self.showed = False

    def show(self):
        self.showed = True
        return self


class MockPlottable:
    def __init__(self):
        self.showed = False

    def show(self):
        self.showed = True
        return self


# ACF/PACF
class MockTSAPlots:
    plot_acf = plot_pacf = (lambda **kwargs: MockPlottable())


# TODO: can we get this to work eventually?
if False:
    @pytest.mark.parametrize('show', [True, False])
    def test_visualizations(show):
        with patch('statsmodels.graphics.tsaplots', MockTSAPlots):

            # Have to import AFTER tha patch, since the pm.__init__ will
            # promptly import the visualization suite, which overwrites the
            # patch
            from pmdarima.utils import visualization
            dataset = np.random.RandomState(42).rand(150)

            # ac_plot = pm.autocorr_plot(dataset, show=show)
            acf_plot = visualization.plot_acf(dataset, show=show)
            pacf_plot = visualization.plot_pacf(dataset, show=show)

            # assert ac_plot.showed is show
            assert acf_plot.showed is show
            assert pacf_plot.showed is show
