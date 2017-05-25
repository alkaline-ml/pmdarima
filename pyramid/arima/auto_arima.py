# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Automatically find optimal parameters for an ARIMA

from __future__ import absolute_import
from sklearn.utils.validation import column_or_1d
from sklearn.externals.joblib import Parallel, delayed
from .utils import get_callable
from .arima import ARIMA
from .stationarity import is_constant, VALID_TESTS
import warnings

__all__ = [
    'auto_arima'
]


def auto_arima(x, start_p=2, d=None, start_q=2, start_P=1, D=None, start_Q=1, max_p=5, max_d=2, max_q=5,
               max_P=2, max_D=1, max_Q=2, max_order=5, stationary=False, seasonal=True, information_criterion='aic',
               stepwise=True, trace=False, approximation=None, truncate=None, xreg=None, test='kpss',
               seasonal_test='ocsb', allow_drift=True, allow_mean=True, bias_adjust=False, n_jobs=1):
    """The ``AutoARIMA`` function seeks to identify the most optimal parameters for an ``ARIMA`` model, and returns
    a fitted ARIMA model.

    Parameters
    ----------
    x : array-like, shape=(n_samples,)
        The time-series to which to fit an ``ARIMA`` estimator.

    start_p : int, optional (default=2)
        The starting value of ``p``, the order (or number of time lags) of the autoregressive ("AR") model.

    d : int, optional (default=None)
        The order of first-differencing. If None (by default), the value will automatically be
        selected based on the results of the ``test``.

    start_q : int, optional (default=2)
        The starting value of ``q``, the order of the moving-average ("MA") model.

    start_P : int, optional (default=1)
        The starting value of ``P`` in stepwise procedure.

    D : int, optional (default=None)
        The order of seasonal differencing. If None (by default), the value will
        automatically be selected based on the results of the ``test``.

    start_Q : int, optional (default=1)
        The starting value of ``Q`` in stepwise procedure.

    max_p : int, optional (default=5)
        The maximum value of ``p``.

    max_d : int, optional (default=2)
        The maximum value of ``d``, or the maximum number of non-seasonal differences.

    max_q : int, optional (default=5)
        The maximum value of ``q``.

    max_P :

    max_D :

    max_Q :

    max_order :

    stationary :

    seasonal :

    information_criterion :

    stepwise :

    trace :

    approximation :

    truncate :

    xreg :

    test :

    seasonal_test :

    allow_drift :

    allow_mean :

    bias_adjust :

    n_jobs :


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    [2] https://github.com/robjhyndman/forecast/blob/19b0711e554524bf6435b7524517715658c07699/R/arima.R
    [3]
    """
    # only non-stepwise parallel implemented
    if stepwise and n_jobs > 1:
        n_jobs = 1
        warnings.warn("Parallel computation is only implemented when stepwise=False. "
                      "The model will be fit in serial.")

    # ensure that x is a 1d ts
    x = column_or_1d(x)
    length = x.shape[0]

    # is constant? return an arima
    if is_constant(x):
        # fit an arima.... make sure to set ``fixed`` in ARIMA fit
        # todo add the extra args
        return ARIMA(p=0, d=0, q=0, P=0, D=0, Q=0, allow_mean=allow_mean, xreg=xreg,
                     allow_drift=allow_drift, bias_adjust=bias_adjust).fit(x)

    # AICC doesn't work for very small samples
    ic = information_criterion
    if length <= 3:
        ic = 'aic'

    # get the functions
    test = get_callable(test, VALID_TESTS)
    # todo: finish
