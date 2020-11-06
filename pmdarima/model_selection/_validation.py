# -*- coding: utf-8 -*-
"""
Cross-validation for ARIMA and pipeline estimators.
See: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py
"""  # noqa: E501

import numpy as np
import numbers
import warnings
import time
from traceback import format_exception_only

from sklearn import base
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import indexable

from ._split import check_cv
from .. import metrics
from ..utils import check_endog
from ..warnings import ModelFitWarning
from ..compat.sklearn import safe_indexing
from ..compat import pmdarima as pm_compat

__all__ = [
    'cross_validate',
    'cross_val_predict',
    'cross_val_score',
]


_valid_scoring = {
    'mean_absolute_error': mean_absolute_error,
    'mean_squared_error': mean_squared_error,
    'smape': metrics.smape,
}

_valid_averaging = {
    'mean': np.nanmean,
    'median': np.nanmedian,
}


def _check_callables(x, dct, varname):
    if callable(x):
        return x
    if isinstance(x, str):
        try:
            return dct[x]
        except KeyError:
            valid_keys = list(dct.keys())
            raise ValueError('%s can be a callable or a string in %s'
                             % (varname, str(valid_keys)))
    raise TypeError('expected a callable or a string, but got %r (type=%s)'
                    % (x, type(x)))


def _check_averaging(method):
    return _check_callables(method, _valid_averaging, "averaging")


def _check_scoring(metric):
    return _check_callables(metric, _valid_scoring, "metric")


def _safe_split(y, X, train, test):
    """Performs the CV indexing given the indices"""
    y_train, y_test = y.take(train), y.take(test)
    if X is None:
        X_train = X_test = None
    else:
        X_train, X_test = safe_indexing(X, train), safe_indexing(X, test)
    return y_train, y_test, X_train, X_test


def _fit_and_score(fold, estimator, y, X, scorer, train, test, verbose,
                   error_score):
    """Fit estimator and compute scores for a given dataset split."""
    msg = 'fold=%i' % fold
    if verbose > 1:
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    start_time = time.time()
    y_train, y_test, X_train, X_test = _safe_split(y, X, train, test)

    try:
        estimator.fit(y_train, X=X_train)

    except Exception as e:
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        else:
            test_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test "
                          "partition will be set to %f. Details: \n%s"
                          % (error_score,
                             format_exception_only(type(e), e)[0]),
                          ModelFitWarning)

    else:
        fit_time = time.time() - start_time

        # forecast h periods into the future and compute the score
        preds = estimator.predict(n_periods=len(test), X=X_test)
        test_scores = scorer(y_test, preds)
        score_time = time.time() - start_time - fit_time

    if verbose > 2:
        total_time = score_time + fit_time
        msg += ", score=%.3f [time=%.3f sec]" % (test_scores, total_time)
        print(msg)

    # TODO: if we ever want train scores, we'll need to change this signature
    return test_scores, fit_time, score_time


def _fit_and_predict(fold, estimator, y, X, train, test, verbose):
    """Fit estimator and compute scores for a given dataset split."""
    msg = 'fold=%i' % fold
    if verbose > 1:
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    start_time = time.time()
    y_train, _, X_train, X_test = _safe_split(y, X, train, test)

    # scikit doesn't handle failures on cv predict, so we won't either.
    estimator.fit(y_train, X=X_train)
    fit_time = time.time() - start_time

    # forecast h periods into the future
    start_time = time.time()
    preds = estimator.predict(n_periods=len(test), X=X_test)
    pred_time = time.time() - start_time

    if verbose > 2:
        total_time = pred_time + fit_time
        msg += " [time=%.3f sec]" % (total_time)
        print(msg)

    return preds, test


def cross_validate(estimator,
                   y,
                   X=None,
                   scoring=None,
                   cv=None,
                   verbose=0,
                   error_score=np.nan,
                   **kwargs):  # TODO: remove kwargs
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Parameters
    ----------
    estimator : estimator
        An estimator object that implements the ``fit`` method

    y : array-like or iterable, shape=(n_samples,)
            The time-series array.

    X : array-like, shape=[n_obs, n_vars], optional (default=None)
        An optional 2-d array of exogenous variables.

    scoring : str or callable, optional (default=None)
        The scoring metric to use. If a callable, must adhere to the signature
        ``metric(true, predicted)``. Valid string scoring metrics include:

        - 'smape'
        - 'mean_absolute_error'
        - 'mean_squared_error'

    cv : BaseTSCrossValidator or None, optional (default=None)
        An instance of cross-validation. If None, will use a RollingForecastCV

    verbose : integer, optional
        The verbosity level.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, ModelFitWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
    """
    # Temporary shim until we remove `exogenous` support completely
    X, _ = pm_compat.get_X(X, **kwargs)

    y, X = indexable(y, X)
    y = check_endog(y, copy=False)

    cv = check_cv(cv)
    scoring = _check_scoring(scoring)

    # validate the error score
    if not (error_score == "raise" or isinstance(error_score, numbers.Number)):
        raise ValueError('error_score should be the string "raise" or a '
                         'numeric value')

    # TODO: in the future we might consider joblib for parallelizing, but it
    #   . could cause cross threads in parallelism..

    results = [
        _fit_and_score(fold,
                       base.clone(estimator),
                       y,
                       X,
                       scorer=scoring,
                       train=train,
                       test=test,
                       verbose=verbose,
                       error_score=error_score)
        for fold, (train, test) in enumerate(cv.split(y, X))]
    scores, fit_times, score_times = list(zip(*results))

    ret = {
        'test_score': np.array(scores),
        'fit_time': np.array(fit_times),
        'score_time': np.array(score_times),
    }
    return ret


def cross_val_predict(estimator,
                      y,
                      X=None,
                      cv=None,
                      verbose=0,
                      averaging="mean",
                      **kwargs):  # TODO: remove kwargs
    """Generate cross-validated estimates for each input data point

    Parameters
    ----------
    estimator : estimator
        An estimator object that implements the ``fit`` method

    y : array-like or iterable, shape=(n_samples,)
            The time-series array.

    X : array-like, shape=[n_obs, n_vars], optional (default=None)
        An optional 2-d array of exogenous variables.

    cv : BaseTSCrossValidator or None, optional (default=None)
        An instance of cross-validation. If None, will use a RollingForecastCV.
        Note that for cross-validation predictions, the CV step cannot exceed
        the CV horizon, or there will be a gap between fold predictions.

    verbose : integer, optional
        The verbosity level.

    averaging : str or callable, one of ["median", "mean"] (default="mean")
        Unlike normal CV, time series CV might have different folds (windows)
        forecasting the same time step. After all forecast windows are made,
        we build a matrix of y x n_folds, populating each fold's forecasts like
        so::

            nan nan nan  # training samples
            nan nan nan
            nan nan nan
            nan nan nan
              1 nan nan  # test samples
              4   3 nan
              3 2.5 3.5
            nan   6   5
            nan nan   4

        We then average each time step's forecasts to end up with our final
        prediction results.

    Examples
    --------
    >>> import pmdarima as pm
    >>> from pmdarima.model_selection import cross_val_predict,\
    ...     RollingForecastCV
    >>> y = pm.datasets.load_wineind()
    >>> cv = RollingForecastCV(h=14, step=12)
    >>> preds = cross_val_predict(
    ...     pm.ARIMA((1, 1, 2), seasonal_order=(0, 1, 1, 12)), y, cv=cv)
    >>> preds[:5]
    array([30710.45743168, 34902.94929722, 17994.16587163, 22127.71167249,
           25473.60876435])
    """
    # Temporary shim until we remove `exogenous` support completely
    X, _ = pm_compat.get_X(X, **kwargs)

    y, X = indexable(y, X)
    y = check_endog(y, copy=False)
    cv = check_cv(cv)
    avgfunc = _check_averaging(averaging)

    # need to be careful here:
    # >>> cv = RollingForecastCV(step=6, h=4)
    # >>> cv_generator = cv.split(wineind)
    # >>> next(cv_generator)
    # (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    #         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    #         45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57]),
    #  array([58, 59, 60, 61]))
    # >>> next(cv_generator)
    # (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
    #         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    #         45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    #         60, 61, 62, 63]),
    #  array([64, 65, 66, 67]))  <~~ 64 vs. 61
    if cv.step > cv.horizon:
        raise ValueError("CV step cannot be > CV horizon, or there will be a "
                         "gap in predictions between folds")

    # clone estimator to make sure all folds are independent
    prediction_blocks = [
        _fit_and_predict(fold,
                         base.clone(estimator),
                         y,
                         X,
                         train=train,
                         test=test,
                         verbose=verbose,)  # TODO: fit params?
        for fold, (train, test) in enumerate(cv.split(y, X))]

    # Unlike normal CV, time series CV might have different folds (windows)
    # forecasting the same time step. In this stage, we build a matrix of
    # y x n_folds, populating each fold's forecasts like so:

    pred_matrix = np.ones((y.shape[0], len(prediction_blocks))) * np.nan
    for i, (pred_block, test_indices) in enumerate(prediction_blocks):
        pred_matrix[test_indices, i] = pred_block

    # from there, we need to apply nanmean (or some other metric) along rows
    # to agree on a forecast for a sample.
    test_mask = ~(np.isnan(pred_matrix).all(axis=1))
    predictions = pred_matrix[test_mask]
    return avgfunc(predictions, axis=1)


def cross_val_score(estimator,
                    y,
                    X=None,
                    scoring=None,
                    cv=None,
                    verbose=0,
                    error_score=np.nan,
                    **kwargs):  # TODO: remove kwargs
    """Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator
        An estimator object that implements the ``fit`` method

    y : array-like or iterable, shape=(n_samples,)
            The time-series array.

    X : array-like, shape=[n_obs, n_vars], optional (default=None)
        An optional 2-d array of exogenous variables.

    scoring : str or callable, optional (default=None)
        The scoring metric to use. If a callable, must adhere to the signature
        ``metric(true, predicted)``. Valid string scoring metrics include:

        - 'smape'
        - 'mean_absolute_error'
        - 'mean_squared_error'

    cv : BaseTSCrossValidator or None, optional (default=None)
        An instance of cross-validation. If None, will use a RollingForecastCV

    verbose : integer, optional
        The verbosity level.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, ModelFitWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
    """
    # Temporary shim until we remove `exogenous` support completely
    X, _ = pm_compat.get_X(X, **kwargs)

    cv_results = cross_validate(estimator=estimator,
                                y=y,
                                X=X,
                                scoring=scoring,
                                cv=cv,
                                verbose=verbose,
                                error_score=error_score)
    return cv_results['test_score']
