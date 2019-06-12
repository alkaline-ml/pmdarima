# -*- coding: utf-8 -*-

import six

import abc

from ..base import BaseTransformer


class BaseExogTransformer(six.with_metaclass(abc.ABCMeta, BaseTransformer)):
    """A base class for exogenous array transformers"""

    def _check_y_exog(self, y, exog, null_allowed=False):
        """Check the endog and exog arrays"""
        y, exog = super(BaseExogTransformer, self)._check_y_exog(y, exog)
        if exog is None and not null_allowed:
            raise ValueError("exog must be non-None for exogenous "
                             "transformers")
        return y, exog


class BaseExogFeaturizer(six.with_metaclass(abc.ABCMeta, BaseExogTransformer)):
    """Exogenous transformers that create exog features from the endog array"""

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
