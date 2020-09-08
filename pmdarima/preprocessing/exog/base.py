# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import abc

from ..base import BaseTransformer


class BaseExogTransformer(BaseTransformer, metaclass=abc.ABCMeta):
    """A base class for exogenous array transformers"""

    def _check_y_X(self, y, X, null_allowed=False):
        """Check the endog and exog arrays"""
        y, X = super(BaseExogTransformer, self)._check_y_X(y, X)
        if X is None and not null_allowed:
            raise ValueError("X must be non-None for exog transformers")
        return y, X


class BaseExogFeaturizer(BaseExogTransformer, metaclass=abc.ABCMeta):
    """Transformers that create new exog features from the endog or exog array

    Parameters
    ----------
    prefix : str or None, optional (default=None)
        The feature prefix
    """
    def __init__(self, prefix=None):
        self.prefix = prefix

    @abc.abstractmethod
    def _get_prefix(self):
        """Get the feature prefix for when exog is a pd.DataFrame"""

    def _get_feature_names(self, X):
        pfx = self._get_prefix()
        return ['%s_%i' % (pfx, i) for i in range(X.shape[1])]

    def _safe_hstack(self, X, features):
        """H-stack dataframes or np.ndarrays"""
        if X is None or isinstance(X, pd.DataFrame):
            # the features we're adding may be np.ndarray
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame.from_records(features)

            # subclass may override this
            features.columns = self._get_feature_names(features)

            if X is not None:
                # ignore_index will remove names, which is a stupid quirk
                # of pandas... so manually reset the indices
                # https://stackoverflow.com/a/43406062/3015734
                X.index = features.index = np.arange(X.shape[0])
                return pd.concat([X, features], axis=1)
            # if X was None coming in, we'd still like to favor a pd.DF
            return features

        return np.hstack([X, features])

    def transform(self, y, X=None, n_periods=0, **kwargs):
        """Transform the new array

        Apply the transformation to the array after learning the training set's
        characteristics in the ``fit`` method. The transform method for
        featurizers behaves slightly differently in that the ``n_periods` may
        be required to extrapolate for periods in the future.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        X : array-like or None, shape=(n_samples, n_features)
            An array of additional covariates.

        n_periods : int, optional (default=0)
            The number of periods in the future to forecast. If ``n_periods``
            is 0, will compute the features for the training set.
            ``n_periods`` corresponds to the number of samples that will be
            returned.

        **kwargs : keyword args
            Keyword arguments required by the transform function.

        Returns
        -------
        y : array-like or None
            The transformed y array

        X : array-like or None
            The transformed X array
        """
