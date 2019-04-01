# -*- coding: utf-8 -*-
#
# Fourier transformations on endogenous arrays

from scipy.fftpack import rfft, irfft

from .base import BaseEndogTransformer

__all__ = ['FourierEndogTransformer']


class FourierEndogTransformer(BaseEndogTransformer):
    """Apply a discrete Fourier transform of a real sequence to an endog array

    The discrete Fourier transform is an invertible, linear equation that
    converts a finite sequence of equally-spaced samples of a function into a
    same-length sequence of equally-spaced samples of the discrete-time Fourier
    transform (DTFT).

    Parameters
    ----------
    n : int, optional
        Length of the Fourier transform. If ``n < y.shape[axis]``, ``y`` is
        truncated. If ``n > y.shape[axis]``, ``y`` is zero-padded. (Default
        ``n = y.shape[axis]``).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Discrete_Fourier_transform
    """
    def __init__(self, n=None):
        self.n = n

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
        # This is largely a pass-though, since the result of the Fourier
        # transform is simply a linear function.
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
        y, exog = self._check_y_exog(y, exog)
        return rfft(y, self.n, axis=-1), exog

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
        y, exog = self._check_y_exog(y, exog)
        return irfft(y, self.n, axis=-1), exog
