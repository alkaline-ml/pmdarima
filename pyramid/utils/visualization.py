# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Plotting wrapper functions

from __future__ import absolute_import

from ..compat.pandas import autocorrelation_plot as ap

from statsmodels.graphics.tsaplots import plot_acf as pacf, plot_pacf as ppacf
from matplotlib import pyplot as plt

__all__ = [
    'autocorr_plot',
    'plot_acf',
    'plot_pacf'
]


def autocorr_plot(series):
    """Plot a series' auto-correlation.

    A wrapper method for the Pandas ``autocorrelation_plot`` method.

    Parameters
    ----------
    series : array-like, shape=(n_samples,)
        The series or numpy array for which to plot an auto-correlation.
    """
    ap(series)
    plt.show()


def plot_acf(series, ax=None, lags=None, alpha=None, use_vlines=True,
             unbiased=False, fft=True, title='Autocorrelation',
             zero=True, vlines_kwargs=None, **kwargs):
    """Plot a series' auto-correlation as a line plot.

    A wrapper method for the statsmodels ``plot_acf`` method.

    Parameters
    ----------
    series : array-like, shape=(n_samples,)
        The series or numpy array for which to plot an auto-correlation.

    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    lags : int, array-like or None, optional (default=None)
        int or Array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.

    alpha : scalar, optional (default=None)
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula. If None, no confidence intervals are plotted.

    use_vlines : bool, optional (default=True)
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.

    unbiased : bool, optional (default=False)
        If True, then denominators for autocovariance are n-k, otherwise n

    fft : bool, optional (default=True)
        If True, computes the ACF via FFT.

    title : str, optional (default='Autocorrelation')
        Title to place on plot. Default is 'Autocorrelation'

    zero : bool, optional (default=True)
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.

    vlines_kwargs : dict, optional (default=None)
        Optional dictionary of keyword arguments that are passed to vlines.

    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.
    """
    pacf(x=series, ax=ax, lags=lags, alpha=alpha, use_vlines=use_vlines,
         unbiased=unbiased, fft=fft, title=title, zero=zero,
         vlines_kwargs=vlines_kwargs, **kwargs)
    plt.show()


def plot_pacf(series, ax=None, lags=None, alpha=None, method='yw',
              use_vlines=True, unbiased=False, fft=True,
              title='Partial Autocorrelation', zero=True,
              vlines_kwargs=None, **kwargs):
    """Plot a series' partial auto-correlation as a line plot.

    A wrapper method for the statsmodels ``plot_pacf`` method.

    Parameters
    ----------
    series : array-like, shape=(n_samples,)
        The series or numpy array for which to plot an auto-correlation.

    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.

    lags : int, array-like or None, optional (default=None)
        int or Array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.

    alpha : scalar, optional (default=None)
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula. If None, no confidence intervals are plotted.

    method : str, optional (default='yw')
        Specifies which method for the calculations to use. One of
        {'ywunbiased', 'ywmle', 'ols', 'ld', 'ldb', 'ldunbiased', 'ldbiased'}:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf. Default.
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction

    use_vlines : bool, optional (default=True)
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.

    unbiased : bool, optional (default=False)
        If True, then denominators for autocovariance are n-k, otherwise n

    fft : bool, optional (default=True)
        If True, computes the ACF via FFT.

    title : str, optional (default='Partial Autocorrelation')
        Title to place on plot. Default is 'Partial Autocorrelation'

    zero : bool, optional (default=True)
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.

    vlines_kwargs : dict, optional (default=None)
        Optional dictionary of keyword arguments that are passed to vlines.

    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.
    """
    ppacf(x=series, ax=ax, lags=lags, alpha=alpha, method=method,
          use_vlines=use_vlines, unbiased=unbiased, fft=fft, title=title,
          zero=zero, vlines_kwargs=vlines_kwargs, **kwargs)
    plt.show()
