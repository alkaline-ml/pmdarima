# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import abc

from ..base import BaseTransformer


class BaseExogTransformer(BaseTransformer, metaclass=abc.ABCMeta):
    """A base class for exogenous array transformers"""

    def _check_y_exog(self, y, exog, null_allowed=False):
        """Check the endog and exog arrays"""
        y, exog = super(BaseExogTransformer, self)._check_y_exog(y, exog)
        if exog is None and not null_allowed:
            raise ValueError("exog must be non-None for exogenous "
                             "transformers")
        return y, exog


class BaseExogFeaturizer(BaseExogTransformer, metaclass=abc.ABCMeta):
    """Exogenous transformers that create exog features from the endog array

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

    def _safe_hstack(self, exog, features):
        """H-stack dataframes or np.ndarrays"""
        if exog is None or isinstance(exog, pd.DataFrame):
            # the features we're adding may be np.ndarray
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame.from_records(features)

            # subclass may override this
            features.columns = self._get_feature_names(features)

            if exog is not None:
                # ignore_index will remove names, which is a stupid quirk
                # of pandas... so manually reset the indices
                # https://stackoverflow.com/a/43406062/3015734
                exog.index = features.index = np.arange(exog.shape[0])
                return pd.concat([exog, features], axis=1)
            # if exog was None coming in, we'd still like to favor a pd.DF
            return features

        return np.hstack([exog, features])

    def transform(self, y, exogenous=None, n_periods=0, **kwargs):
        """Transform the new array

        Apply the transformation to the array after learning the training set's
        characteristics in the ``fit`` method. The transform method for
        featurizers behaves slightly differently in that the ``n_periods` may
        be required to extrapolate for periods in the future.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features)
            The exogenous array of additional covariates.

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

        exogenous : array-like or None
            The transformed exogenous array
        """
