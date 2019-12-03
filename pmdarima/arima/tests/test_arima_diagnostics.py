# -*- coding: utf-8 -*-

from pmdarima.datasets import load_lynx
from pmdarima.arima import ARIMA

from unittest.mock import patch
import pytest

lynx = load_lynx()


class MockMPLFigure:
    def __init__(self, fig, figsize):
        self.fig = fig
        self.figsize = figsize
        self.subplots = []

    def add_subplot(self, *args):
        ax = MockMPLAxis(*args)
        self.subplots.append(ax)
        return ax


class MockMPLAxis:
    def __init__(self, *args):
        pass

    def hist(self, *args, **kwargs):
        pass

    def hlines(self, *args, **kwargs):
        # We can hack our assertion here since we always pass alpha=0.5
        for k, v in kwargs.items():
            setattr(self, k, v)

    def legend(self):
        pass

    def plot(self, x, y, **kwargs):
        self.x = x
        self.y = y

    def set_title(self, title):
        self.title = title

    def set_xlim(self, *args):
        if len(args) == 2:
            mn, mx = args
        else:  # len(args) == 1
            mn, mx = args[0]

        self.mn = mn
        self.mx = mx

    def set_ylim(self, mn, mx):
        self.mn = mn
        self.mx = mx


def mock_qqplot(resid, line, ax):
    ax.qqplot_called = True


def mock_acf_plot(resid, ax, lags):
    ax.acfplot_called = True


@pytest.mark.parametrize(
    'model_type,model', [
        pytest.param('arma', ARIMA(order=(1, 0, 0), maxiter=50)),
        pytest.param('arima', ARIMA(order=(1, 1, 0), maxiter=50)),
        pytest.param('sarimax', ARIMA(order=(1, 1, 0),
                                      maxiter=50,
                                      seasonal_order=(1, 0, 0, 12)))
    ])
def test_mock_plot_diagnostics(model_type, model):
    model.fit(lynx)

    with patch('statsmodels.graphics.utils.create_mpl_fig', MockMPLFigure),\
            patch('statsmodels.graphics.gofplots.qqplot', mock_qqplot),\
            patch('statsmodels.graphics.tsaplots.plot_acf', mock_acf_plot):

        diag = model.plot_diagnostics(figsize=(10, 12))

        # Asserting on mock attributes to show that we follow the expected
        # logical branches
        assert diag.figsize == (10, 12)
        assert len(diag.subplots) == 4

        # First one should have 'alpha' from the plot call
        assert hasattr(diag.subplots[0], 'alpha') and \
            diag.subplots[0].alpha == 0.5

        # Third figure gets QQPLOT called on it
        assert hasattr(diag.subplots[2], 'qqplot_called') and \
            diag.subplots[2].qqplot_called

        # Fourth figure gets ACF plot call on it
        assert hasattr(diag.subplots[3], 'acfplot_called') and \
            diag.subplots[3].acfplot_called
