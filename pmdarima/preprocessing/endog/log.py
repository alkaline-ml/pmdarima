# -*- coding: utf-8 -*-

from .boxcox import BoxCoxEndogTransformer

__all__ = ['LogEndogTransformer']


class LogEndogTransformer(BoxCoxEndogTransformer):
    """Apply a log transformation to an endogenous array

    When ``y`` is your endogenous array, the log transform is
    ``log(y + lmbda)``

    Parameters
    ----------

    lmbda : float, optional (default=0.)
        The value to add to ``y`` to make it non-negative. If, after adding
        ``lmbda``, there are still negative values, a ValueError will be
        raised.

    neg_action : str, optional (default="raise")
        How to respond if any values in ``y <= 0`` after adding ``lmbda``.
        One of ('raise', 'warn', 'ignore'). If anything other than 'raise',
        values <= 0 will be truncated to the value of ``floor``.

    floor : float, optional (default=1e-16)
        A positive value that truncate values to if there are values in ``y``
        that are zero or negative and ``neg_action`` is not 'raise'. Note that
        if values are truncated, invertibility will not be preserved, and the
        transformed array may not be perfectly inverse-transformed.
    """
    def __init__(self, lmbda=0, neg_action="raise", floor=1e-16):

        super().__init__(0, lmbda2=lmbda, neg_action=neg_action, floor=floor)

    def fit(self, y, exogenous=None):
        """Fit the transformer

        Must be called before ``transform``.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.
        """
        return super().fit(y, exogenous)

    def transform(self, y, exogenous=None, **transform_kwargs):
        """Apply the log transform to the array

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        y_transform : array-like or None
            The log transformed y array

        exogenous : array-like or None
            The exog array
        """
        return super().transform(y, exogenous, **transform_kwargs)

    def inverse_transform(self, y, exogenous=None):
        """Inverse transform a transformed array

        Inverse the log transformation on the transformed array. Note that
        if truncation happened in the ``transform`` method, invertibility will
        not be preserved, and the transformed array may not be perfectly
        inverse-transformed.

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
        return super().inverse_transform(y, exogenous=exogenous)
