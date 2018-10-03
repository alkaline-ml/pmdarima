# -*- coding: utf-8 -*-
#
# Author: Taylor G Smith <taylor.smith@alkaline-ml.com>
#
# Plotting wrapper functions

from __future__ import absolute_import

from ..compat.pandas import autocorrelation_plot as ap
from ..compat.matplotlib import get_compatible_pyplot

from statsmodels.graphics.tsaplots import plot_acf as pacf, plot_pacf as ppacf

import warnings
import os

# User may not have matplotlib
try:
    # Gets the MPL.pyplot import (combatibilitized). Only use debug mode if set
    # on the machine in an environment variable
    debug = os.environ.get("PYRAMID_MPL_DEBUG", "false").lower() == "true"

    # If it's a Travis CI machine, we want to set the backend via env variable
    backend = os.environ.get("PYRAMID_MPL_BACKEND", None)
    plt = get_compatible_pyplot(backend=backend, debug=debug)
except ImportError:
    plt = None

__all__ = [
    'autocorr_plot',
    'plot_acf',
    'plot_pacf'
]


def warn_for_no_mpl():
    if plt is None:
        warnings.warn("You do not have matplotlib installed. In order to "
                      "create plots, you'll need to pip install matplotlib!")


def _show_or_return(obj, show):
    if show:
        # We never cover this in tests, unfortunately. Even with the
        # cleanup tag, Travis doesn't play super nice with showing and
        # closing lots of plots over and over. But it's just one line...
        plt.show()
        # returns None implicitly
    else:
        return obj


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
    if plt is None:
        warn_for_no_mpl()
        return None

    res = ap(series)
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
    if plt is None:
        warn_for_no_mpl()
        return None

    res = pacf(x=series, ax=ax, lags=lags, alpha=alpha, use_vlines=use_vlines,
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
    if plt is None:
        warn_for_no_mpl()
        return None

    res = ppacf(x=series, ax=ax, lags=lags, alpha=alpha, method=method,
                use_vlines=use_vlines, title=title, zero=zero,
                vlines_kwargs=vlines_kwargs, **kwargs)

    return _show_or_return(res, show)
