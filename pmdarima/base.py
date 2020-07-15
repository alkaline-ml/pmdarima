# -*- coding: utf-8 -*-
#
# Base classes and interfaces

import abc
from abc import ABCMeta

from sklearn.base import BaseEstimator

# TODO: change this to base TS model if we ever hope to support more


class BaseARIMA(BaseEstimator, metaclass=ABCMeta):
    """A base ARIMA class"""

    @abc.abstractmethod
    def fit(self, y, exogenous, **fit_args):
        """Fit an ARIMA model"""

    def fit_predict(self, y, exogenous=None, n_periods=10, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations with an
        optional matrix of ``exogenous`` variables, and then generate
        predictions.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series to which to fit the ``ARIMA`` estimator. This may
            either be a Pandas ``Series`` object (statsmodels can internally
            use the dates in the index), or a numpy array. This should be a
            one-dimensional array of floats, and should not contain any
            ``np.nan`` or ``np.inf`` values.

        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        fit_args : dict or kwargs, optional (default=None)
            Any keyword args to pass to the fit method.
        """
        self.fit(y, exogenous, **fit_args)
        return self.predict(n_periods=n_periods, exogenous=exogenous)

    @abc.abstractmethod
    def predict(self, n_periods, exogenous, return_conf_int=False, alpha=0.05):
        """Create forecasts on a fitted model"""

    @abc.abstractmethod
    def predict_in_sample(self, exogenous, start, end, dynamic):
        """Get in-sample forecasts"""

    @abc.abstractmethod
    def update(self, y, exogenous=None, maxiter=None, **kwargs):
        """Update an ARIMA model"""
