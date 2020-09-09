# -*- coding: utf-8 -*-

from scipy import stats

import numpy as np
import warnings

from ...compat import check_is_fitted, pmdarima as pm_compat
from .base import BaseEndogTransformer

__all__ = ['BoxCoxEndogTransformer']


class BoxCoxEndogTransformer(BaseEndogTransformer):
    r"""Apply the Box-Cox transformation to an endogenous array

    The Box-Cox transformation is applied to non-normal data to coerce it more
    towards a normal distribution. It's specified as::

        (((y + lam2) ** lam1) - 1) / lam1, if lmbda != 0, else
        log(y + lam2)

    Parameters
    ----------
    lmbda : float or None, optional (default=None)
        The lambda value for the Box-Cox transformation, if known. If not
        specified, it will be estimated via MLE.

    lmbda2 : float, optional (default=0.)
        The value to add to ``y`` to make it non-negative. If, after adding
        ``lmbda2``, there are still negative values, a ValueError will be
        raised.

    neg_action : str, optional (default="raise")
        How to respond if any values in ``y <= 0`` after adding ``lmbda2``.
        One of ('raise', 'warn', 'ignore'). If anything other than 'raise',
        values <= 0 will be truncated to the value of ``floor``.

    floor : float, optional (default=1e-16)
        A positive value that truncate values to if there are values in ``y``
        that are zero or negative and ``neg_action`` is not 'raise'. Note that
        if values are truncated, invertibility will not be preserved, and the
        transformed array may not be perfectly inverse-transformed.
    """
    def __init__(self, lmbda=None, lmbda2=0, neg_action="raise", floor=1e-16):

        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.neg_action = neg_action
        self.floor = floor

    def fit(self, y, X=None, **kwargs):  # TODO: kwargs go away
        """Fit the transformer

        Learns the value of ``lmbda``, if not specified in the constructor.
        If defined in the constructor, is not re-learned.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        X : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.
        """
        lam1 = self.lmbda
        lam2 = self.lmbda2

        # Temporary shim until we remove `exogenous` support completely
        X, _ = pm_compat.get_X(X, **kwargs)

        if lam2 < 0:
            raise ValueError("lmbda2 must be a non-negative scalar value")

        if lam1 is None:
            y, _ = self._check_y_X(y, X)
            _, lam1 = stats.boxcox(y + lam2, lmbda=None, alpha=None)

        self.lam1_ = lam1
        self.lam2_ = lam2
        return self

    def transform(self, y, X=None, **kwargs):
        """Transform the new array

        Apply the Box-Cox transformation to the array after learning the
        lambda parameter.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        X : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        y_transform : array-like or None
            The Box-Cox transformed y array

        X : array-like or None
            The X array
        """
        check_is_fitted(self, "lam1_")

        # Temporary shim until we remove `exogenous` support completely
        X, _ = pm_compat.get_X(X, **kwargs)

        lam1 = self.lam1_
        lam2 = self.lam2_

        y, exog = self._check_y_X(y, X)
        y += lam2

        neg_mask = y <= 0.
        if neg_mask.any():
            action = self.neg_action
            msg = "Negative or zero values present in y"
            if action == "raise":
                raise ValueError(msg)
            elif action == "warn":
                warnings.warn(msg, UserWarning)
            y[neg_mask] = self.floor

        if lam1 == 0:
            return np.log(y), exog
        return (y ** lam1 - 1) / lam1, exog

    def inverse_transform(self, y, X=None, **kwargs):  # TODO: kwargs go away
        """Inverse transform a transformed array

        Inverse the Box-Cox transformation on the transformed array. Note that
        if truncation happened in the ``transform`` method, invertibility will
        not be preserved, and the transformed array may not be perfectly
        inverse-transformed.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The transformed endogenous (time-series) array.

        X : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        y : array-like or None
            The inverse-transformed y array

        X : array-like or None
            The inverse-transformed X array
        """
        check_is_fitted(self, "lam1_")

        # Temporary shim until we remove `exogenous` support completely
        X, _ = pm_compat.get_X(X, **kwargs)

        lam1 = self.lam1_
        lam2 = self.lam2_

        y, exog = self._check_y_X(y, X)
        if lam1 == 0:
            return np.exp(y) - lam2, exog

        numer = y * lam1  # remove denominator
        numer += 1.  # add 1 back to it
        de_exp = numer ** (1. / lam1)  # de-exponentiate
        return de_exp - lam2, exog
