# -*- coding: utf-8 -*-

import abc

from ..base import BaseTransformer


class BaseEndogTransformer(BaseTransformer, metaclass=abc.ABCMeta):
    """A base class for endogenous array transformers"""

    def _check_y_exog(self, y, exog):
        """Check the endog and exog arrays"""
        y, exog = super(BaseEndogTransformer, self)._check_y_exog(y, exog)
        if y is None:
            raise ValueError("y must be non-None for endogenous transformers")
        return y, exog

    @abc.abstractmethod
    def inverse_transform(self, y, exogenous=None):
        """Inverse transform a transformed array

        Inverse the transformation on the transformed array.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The transformed endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        y : array-like or None
            The inverse-transformed y array

        exogenous : array-like or None
            The inverse-transformed exogenous array
        """
