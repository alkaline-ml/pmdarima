# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Plotting wrapper functions

from .array import check_endog
from ..compat.pandas import plotting as pd_plotting
from ..compat.matplotlib import get_compatible_pyplot

from statsmodels.graphics import tsaplots

import numpy as np
import os

# User may not have matplotlib
try:
    # Gets the MPL.pyplot import (combatibilitized). Only use debug mode if set
    # on the machine in an environment variable
    debug = os.environ.get("PMDARIMA_MPL_DEBUG", "false").lower() == "true"

    # If it's a Travis CI machine, we want to set the backend via env variable
    backend = os.environ.get("PMD_MPL_BACKEND", None)
    mpl = get_compatible_pyplot(backend=backend, debug=debug)

except ImportError:
    mpl = None

__all__ = [
    'autocorr_plot',
    'decomposed_plot',
    'plot_acf',
    'plot_pacf',
    'tsdisplay',
]


def _err_for_no_mpl():
    if mpl is None:
        # Per Issue #47:
        raise ImportError(
            "You do not have matplotlib installed. In order to "
            "create plots, you'll need to pip install matplotlib!")


def _get_plt():
    """Get MPL pyplot if it exists or raise an error if not"""
    _err_for_no_mpl()
    return mpl


def _show_or_return(obj, show):
    if show:
        # We never cover this in tests, unfortunately. Even with the
        # cleanup tag, Travis doesn't play super nice with showing and
        # closing lots of plots over and over. But it's just one line...
        mpl.show()
        # returns None implicitly
    else:
        return obj


def decomposed_plot(decomposed_tuple, figure_kwargs=None, show=True):
    """Plot the decomposition of a time series.

    Plots the results of the time series decomposition in four plots:
    the 'x', 'trend', 'seasonal', and 'random' components.

    Parameters
    ----------
    decomposed_tuple : tuple, namedtuple or iterable
        Named tuple of series that consist of data, trend, seasonal, and
        random. Should be the result of :func:`pmdarima.arima.decompose`.

    figure_kwargs : dict, optional (default=None)
        Optional dictionary of keyword arguments that are passed to figure.

    show : bool, optional (default=True)
        Whether to show the plot after it's been created. If not, will return
        the plot as an Axis object instead.

    Notes
    -----
    This method will only show the plot if ``show=True`` (which is the default
    behavior). To simply get the axis back (say, to add to another canvas),
    use ``show=False``.
    """

    _err_for_no_mpl()

    fig, axes = mpl.subplots(4, 1, sharex=True, **figure_kwargs)

    y_labels = ['data', 'trend', 'seasonal', 'random']

    for ax in axes.flat:
        ax.set(ylabel=y_labels.pop(0))

    # Unpack positionally so that user could pass as a tuple, if they have
    # pre-computed...
    x, trend, ssnl, rand = decomposed_tuple

    axes[0].plot(x)
    axes[1].plot(trend)
    axes[2].plot(ssnl)
    axes[3].plot(rand)

    return _show_or_return(axes, show)


def autocorr_plot(series, show=True):
    """Plot a series' auto-correlation.

    A wrapper method for the Pandas ``autocorrelation_plot`` method.

    Parameters
    ----------
    series : array-like, shape=(n_samples,)
        The series or numpy array for which to plot an auto-correlation.

    show : bool, optional (default=True)
        Whether to show the plot after it's been created. If not, will return
        the plot as an Axis object instead.

    Notes
    -----
    This method will only show the plot if ``show=True`` (which is the default
    behavior). To simply get the axis back (say, to add to another canvas),
    use ``show=False``.

    Examples
    --------
    >>> autocorr_plot([1, 2, 3], False)  # doctest: +SKIP
    <matplotlib.axes._subplots.AxesSubplot object at 0x127f41dd8>

    Returns
    -------
    res : Axis or None
        If ``show`` is True, does not return anything. If False, returns
        the Axis object.
    """
    _err_for_no_mpl()
    res = pd_plotting.autocorrelation_plot(series)
    return _show_or_return(res, show)


def plot_acf(series, ax=None, lags=None, alpha=None, use_vlines=True,
             unbiased=False, fft=True, title='Autocorrelation',
             zero=True, vlines_kwargs=None, show=True, **kwargs):
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

    show : bool, optional (default=True)
        Whether to show the plot after it's been created. If not, will return
        the plot as an Axis object instead.

    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Notes
    -----
    This method will only show the plot if ``show=True`` (which is the default
    behavior). To simply get the axis back (say, to add to another canvas),
    use ``show=False``.

    Examples
    --------
    >>> plot_acf([1, 2, 3], show=False)  # doctest: +SKIP
    <matplotlib.figure.Figure object at 0x122fab4e0>

    Returns
    -------
    plt : Axis or None
        If ``show`` is True, does not return anything. If False, returns
        the Axis object.
    """
    _err_for_no_mpl()
    res = tsaplots.plot_acf(
        x=series, ax=ax, lags=lags, alpha=alpha, use_vlines=use_vlines,
        unbiased=unbiased, fft=fft, title=title, zero=zero,
        vlines_kwargs=vlines_kwargs, **kwargs)

    return _show_or_return(res, show)


def plot_pacf(series, ax=None, lags=None, alpha=None, method='yw',
              use_vlines=True, title='Partial Autocorrelation', zero=True,
              vlines_kwargs=None, show=True, **kwargs):
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

    title : str, optional (default='Partial Autocorrelation')
        Title to place on plot. Default is 'Partial Autocorrelation'

    zero : bool, optional (default=True)
        Flag indicating whether to include the 0-lag autocorrelation.
        Default is True.

    vlines_kwargs : dict, optional (default=None)
        Optional dictionary of keyword arguments that are passed to vlines.

    show : bool, optional (default=True)
        Whether to show the plot after it's been created. If not, will return
        the plot as an Axis object instead.

    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.

    Notes
    -----
    This method will only show the plot if ``show=True`` (which is the default
    behavior). To simply get the axis back (say, to add to another canvas),
    use ``show=False``.

    Examples
    --------
    >>> plot_pacf([1, 2, 3, 4], show=False)  # doctest: +SKIP
    <matplotlib.figure.Figure object at 0x129df1630>

    Returns
    -------
    plt : Axis or None
        If ``show`` is True, does not return anything. If False, returns
        the Axis object.
    """
    _err_for_no_mpl()
    res = tsaplots.plot_pacf(
        x=series, ax=ax, lags=lags, alpha=alpha, method=method,
        use_vlines=use_vlines, title=title, zero=zero,
        vlines_kwargs=vlines_kwargs, **kwargs)

    return _show_or_return(res, show)


def tsdisplay(y, lag_max=50, figsize=(8, 6), title=None, bins=25,
              series_kwargs=None, acf_kwargs=None, hist_kwargs=None,
              show=True):
    """Display the time series and some of its key statistics

    The equivalent of R's ``forecast::tsdisplay``, showing the series, the
    histogram and the ACF plot.

    Parameters
    ----------
    y : array-like, shape=(n_samples,)
        The series or numpy array for which to plot an auto-correlation.

    lag_max : int, optional (default=50)
        The number of lags for the ACF plot

    figsize : tuple, optional (default=(8, 6))
        The size of the figure

    title : str, optional (default=None)
        A title for the series, if any.

    bins : int, optional (default=25)
        The number of bins for the histogram

    series_kwargs : dict or None, optional (default=None)
        Keyword arguments to pass when plotting the series

    acf_kwargs : dict or None, optional (default=None)
        Keyword arguments to pass when plotting the ACF

    hist_kwargs : dict or None, optional (default=None)
        Keyword arguments to pass when plotting the histogram

    show : bool, optional (default=True)
        Whether to show the plot after it's been created. If not, will return
        the plot as a Figure object instead.

    Examples
    --------
    >>> import pmdarima as pm
    >>> tsdisplay(pm.datasets.load_sunspots(), show=False)
    <Figure size 800x600 with 3 Axes>

    Returns
    -------
    plt : Figure or None
        If ``show`` is True, does not return anything. If False, returns
        the Figure object.
    """

    _err_for_no_mpl()
    from matplotlib import gridspec

    fig = mpl.figure(figsize=figsize)
    gs = gridspec.GridSpec(4, 2)
    ax0 = fig.add_subplot(gs[0:2, 0:])
    ax1 = fig.add_subplot(gs[2:, 0])
    ax2 = fig.add_subplot(gs[2:, 1])

    # make sure y is a np array
    y = check_endog(y, copy=False, preserve_series=True)

    if lag_max >= y.shape[0]:
        raise ValueError(
            f"lag_max ({lag_max}) must be < length of the "
            f"series ({y.shape[0]})"
        )

    # ax0 is simply the series itself
    x0 = np.arange(y.shape[0])
    xlabs = None
    if hasattr(y, 'index'):
        xlabs = y.index.tolist()
        y = y.values
    series_kwargs = {} if not series_kwargs else series_kwargs
    ax0.plot(x0, y, **series_kwargs)
    if title:
        ax0.set_title(title)

    if xlabs is not None:
        # TODO: eventually would be nice to do some well spaced xticks
        pass

    # ax1 is the ACF, so we can just use our ACF plotting func
    acf_kwargs = {} if not acf_kwargs else acf_kwargs
    plot_acf(y, ax=ax1, show=False, title='ACF', lags=lag_max, **acf_kwargs)

    # ax2 is simply the histogram
    hist_kwargs = {} if not hist_kwargs else hist_kwargs
    _ = ax2.hist(y, bins=bins, **hist_kwargs)  # noqa
    ax2.set_title("Frequency")

    fig.tight_layout()
    return _show_or_return(fig, show)
