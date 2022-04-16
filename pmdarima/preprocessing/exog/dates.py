# -*- coding: utf-8 -*-

from .base import BaseExogFeaturizer

import numpy as np
import pandas as pd

import warnings

__all__ = [
    "DateFeaturizer"
]

# TODO: future usecases might include with_hour_of_day


def _safe_hstack_numpy(left, right):
    if left is None:
        return right
    return np.hstack([left, right])


class DateFeaturizer(BaseExogFeaturizer):
    """Create exogenous date features

    Given an exogenous feature of dtype TimeStamp, creates a set of dummy and
    ordinal variables indicating:

      * Day of the week
          Particular days of the week may align with quasi-seasonal trends.

      * Day of the month
          Useful for modeling things like the end-of-month effect, ie., a
          department spends the remainder of its monthly budget to avoid future
          budget cuts, and the last Friday of the month is heavy on spending.

    The motivation for this featurizer comes from a blog post by Rob Hyndman
    [1] on modeling quasi-seasonal patterns in time series. Note that an
    exogenous array _must_ be provided at inference.

    Parameters
    ----------
    column_name : str
        The name of the date column. This forces the exogenous array to be a
        Pandas DataFrame, and does not permit a np.ndarray as others may.

    with_day_of_week : bool, optional (default=True)
        Whether to include dummy variables for the day of the week (in {0, 1}).

    with_day_of_month : bool, optional (default=True)
        Whether to include an ordinal feature for the day of the month (1-31).

    prefix : str or None, optional (default=None)
        The feature prefix

    Examples
    --------
    >>> from pmdarima.datasets._base import load_date_example
    >>> y, X = load_date_example()
    >>> feat = DateFeaturizer(column_name='date')
    >>> _, X_prime = feat.fit_transform(y, X)
    >>> X_prime.head()
       DATE-WEEKDAY-0  DATE-WEEKDAY-1  ...  DATE-WEEKDAY-6  DATE-DAY-OF-MONTH
    0               0               1  ...               0                  1
    1               0               0  ...               0                  2
    2               0               0  ...               0                  3
    3               0               0  ...               0                  4
    4               0               0  ...               0                  5

    Notes
    -----
    * In order to use time series with holes, it is required that an X
      array be provided at prediction time. Other featurizers automatically
      create exog arrays into the future for inference, but this is not
      possible currently with the date featurizer. Your code must provide the
      dates for which you are forecasting as exog features.

    * The ``column_name`` field is dropped in the transformed exogenous array.

    References
    ----------
    .. [1] https://robjhyndman.com/hyndsight/monthly-seasonality/
    """

    def __init__(self, column_name, with_day_of_week=True,
                 with_day_of_month=True, prefix=None):
        super().__init__(prefix=prefix)

        self.column_name = column_name
        self.with_day_of_week = with_day_of_week
        self.with_day_of_month = with_day_of_month

    def _check_X(self, X):
        # exog must be a pd.DataFrame, and the column_name must be a timestamp
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a DataFrame to use the DateFeaturizer, but got "
                f"type={type(X)}"
            )

        name = self.column_name
        if not (name in X.columns and
                'datetime64' in X[name].dtype.name):
            raise ValueError("column '%s' must exist in exog as a "
                             "pd.Timestamp type"
                             % name)

    def _get_prefix(self):
        pfx = self.prefix
        if pfx is None:
            pfx = "DATE"
        return pfx

    # Overrides super abstract method
    def _get_feature_names(self, X):
        pfx = self._get_prefix()
        out = []

        # Something to note is that in Python, 0 is Monday (not Sunday). See
        # comments here: https://stackoverflow.com/a/9847269/3015734
        # E.g., ['DATE-WEEKDAY-0', 'DATE-WEEKDAY-1', ...]
        if self.with_day_of_week:
            out += ['%s-WEEKDAY-%i' % (pfx, i) for i in range(7)]

        if self.with_day_of_month:
            out += ['%s-DAY-OF-MONTH' % pfx]

        return out

    def fit(self, y, X=None, **kwargs):  # TODO: remove kwargs later
        """Fit the transformer

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        X : array-like, shape=(n_samples, n_features)
            The exogenous array of additional covariates. Must include the
            ``column_name`` feature, which must be a pd.Timestamp dtype.
        """
        y, X = self._check_y_X(y, X, null_allowed=False)

        # enforce pd.DataFrame
        self._check_X(X)

        # we don't _technically_ need to do this, but it seems like a nice bit
        # of friendly validation to make sure that at least _something_ will
        # happen in this transformer.
        if not (self.with_day_of_month or self.with_day_of_week):
            warnings.warn("DateTransformer will have no effect given disabled "
                          "parameters")

        return self

    def transform(self, y, X=None, **kwargs):
        """Create date features

        When an ARIMA is fit with an X array, it must be forecasted
        with one also. However, unlike other exogenous featurizers, an X
        array is required at inference time for the DateFeaturizer.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array. This is unused and technically
            optional for the Fourier terms, since it uses the pre-computed
            ``n`` to calculate the seasonal Fourier terms.

        X : array-like, shape=(n_samples, n_features)
            The exogenous array of additional covariates. The ``column_name``
            feature must be present, and of dtype pd.Timestamp
        """
        y, X = self._check_y_X(y, X, null_allowed=True)

        # enforce pd.DataFrame
        self._check_X(X)
        date_series = X[self.column_name]  # type: pd.Series
        m = X.shape[0]

        # the right side of the exog array out
        right_side = None

        if self.with_day_of_week:
            # we cannot use pd.get_dummies because for a test set with < 7 obs
            # we will not produce all the features we need to. create a matrix
            # of zeros and mask manually
            zeros = np.zeros((m, 7), dtype=int)
            zeros[np.arange(zeros.shape[0]), date_series.dt.weekday.values] = 1
            right_side = zeros

        if self.with_day_of_month:
            day_of_month = date_series.dt.day.values.reshape(-1, 1)
            right_side = _safe_hstack_numpy(right_side, day_of_month)

        # stack along axis 1
        if right_side is not None:
            X = self._safe_hstack(X.drop(self.column_name, axis=1),
                                  right_side)
        return y, X
