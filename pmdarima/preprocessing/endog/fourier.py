# -*- coding: utf-8 -*-
#
# Fourier transformations on endogenous arrays

from .base import BaseEndogTransformer

__all__ = ['FourierEndogTransformer']


class FourierEndogTransformer(BaseEndogTransformer):
    r"""Apply the Fourier transformation to an endogenous array

    Parameters
    ----------
    TODO
    """
    def __init__(self):
        pass

    def fit(self, y, exog=None):
        """Fit the transformer

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exog : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.
        """
        # TODO:
        return self

    def transform(self, y, exog=None):
        """Transform the new array

        Apply the Fourier transformation to the array.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exog : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        y_transform : array-like or None
            The Fourier transformed y array

        exog : array-like or None
            The exog array
        """
        # TODO:

    def inverse_transform(self, y, exog=None):
        """Inverse transform a transformed array

        Invert the Fourier transformation on the transformed array.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The transformed endogenous (time-series) array.

        exog : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        y : array-like or None
            The inverse-transformed y array

        exog : array-like or None
            The inverse-transformed exogenous array
        """
        # TODO:
