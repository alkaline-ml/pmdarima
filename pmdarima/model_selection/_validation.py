# -*- coding: utf-8 -*-
"""
Cross-validation for ARIMA and pipeline estimators.
See: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py  # noqa: E501
"""

import numpy as np
import numbers
import warnings
import time
from traceback import format_exception_only

from sklearn import base
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import indexable, safe_indexing

from ._split import check_cv
from .. import metrics
from ..utils import check_endog
from ..arima.warnings import ModelFitWarning

__all__ = [
    'cross_validate',
    'cross_val_score'
]


_valid_scoring = {
    'mean_absolute_error': mean_absolute_error,
    'mean_squared_error': mean_squared_error,
    'smape': metrics.smape,
}


def _check_scoring(metric):
    if callable(metric):
        return metric
    if isinstance(metric, str):
        try:
            return _valid_scoring[metric]
        except KeyError:
            raise ValueError('metric can be a callable or a string in %s'
                             % str(list(_valid_scoring.keys())))
    raise TypeError('expected a callable or a string, but got %r (type=%s)'
                    % (metric, type(metric)))


def _safe_split(y, exog, train, test):
    """Performs the CV indexing given the indices"""
    y_train, y_test = y.take(train), y.take(test)
    if exog is None:
        exog_train = exog_test = None
    else:
        exog_train, exog_test = \
            safe_indexing(exog, train), safe_indexing(exog, test)
    return y_train, y_test, exog_train, exog_test


def _fit_and_score(fold, estimator, y, exog, scorer, train, test, verbose,
                   error_score):
    """Fit estimator and compute scores for a given dataset split."""
    msg = 'fold=%i' % fold
    if verbose > 1:
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    start_time = time.time()
    y_train, y_test, exog_train, exog_test = _safe_split(y, exog, train, test)

    try:
        estimator.fit(y_train, exogenous=exog_train)

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
        preds = estimator.predict(n_periods=len(test), exogenous=exog_test)
        test_scores = scorer(y_test, preds)
        score_time = time.time() - start_time - fit_time

    if verbose > 2:
        total_time = score_time + fit_time
        msg += ", score=%.3f [time=%.3f sec]" % (test_scores, total_time)
        print(msg)

    # TODO: if we ever want train scores, we'll need to change this signature
    return test_scores, fit_time, score_time


def cross_validate(estimator, y, exogenous=None, scoring=None, cv=None,
                   verbose=0, error_score=np.nan):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    Parameters
    ----------
    estimator : estimator
        An estimator object that implements the ``fit`` method

    y : array-like or iterable, shape=(n_samples,)
            The time-series array.

    exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
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
    y, exog = indexable(y, exogenous)
    y = check_endog(y, copy=False)

    cv = check_cv(cv)
    scoring = _check_scoring(scoring)

    # validate the error score
    if not (error_score == "raise" or isinstance(error_score, numbers.Number)):
        raise ValueError('error_score should be the string "raise" or a '
                         'numeric value')

    # TODO: clone between each iteration?
    # TODO: in the future we might consider joblib for parallelizing, but it
    #   . could cause cross threads in parallelism..

    results = [
        _fit_and_score(fold, base.clone(estimator), y, exog,
                       scorer=scoring,
                       train=train,
                       test=test,
                       verbose=verbose,
                       error_score=error_score)
        for fold, (train, test) in enumerate(cv.split(y, exog))]
    scores, fit_times, score_times = list(zip(*results))

    ret = {
        'test_score': np.array(scores),
        'fit_time': np.array(fit_times),
        'score_time': np.array(score_times),
    }
    return ret


def cross_val_score(estimator, y, exogenous=None, scoring=None, cv=None,
                    verbose=0, error_score=np.nan):
    """Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator
        An estimator object that implements the ``fit`` method

    y : array-like or iterable, shape=(n_samples,)
            The time-series array.

    exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
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
    cv_results = cross_validate(estimator=estimator, y=y, exogenous=exogenous,
                                scoring=scoring, cv=cv,
                                verbose=verbose,
                                error_score=error_score)
    return cv_results['test_score']
