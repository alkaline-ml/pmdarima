# -*- coding: utf-8 -*-
#
# Base ARIMA pre-processing classes. Don't import this in __init__, or we'll
# potentially get circular imports in sub-classes

from sklearn.base import BaseEstimator, TransformerMixin
import abc

from ..compat.numpy import DTYPE
from ..utils import check_exog, check_endog

__all__ = [
    "BaseTransformer"
]


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=abc.ABCMeta):
    """A base pre-processing transformer

    A subclass of the scikit-learn ``TransformerMixin``, the purpose of the
    ``BaseTransformer`` is to learn characteristics from the training set and
    apply them in a transformation to the test set. For instance, a transformer
    aimed at normalizing features in an exogenous array would learn the means
    and standard deviations of the training features in the ``fit`` method, and
    then center and scale the features in the ``transform`` method.

    The ``fit`` method should only ever be applied to the *training* set to
    avoid any data leakage, while ``transform`` may be applied to any dataset
    of the same schema.
    """
    @staticmethod
    def _check_y_exog(y, exog):
        """Validate input"""
        # Do not force finite, since a transformer's goal may be imputation.
        if y is not None:
            y = check_endog(y, dtype=DTYPE, copy=True, force_all_finite=False)

        if exog is not None:
            exog = check_exog(
                exog, dtype=None, copy=True, force_all_finite=False)
        return y, exog

    def fit_transform(self, y, exogenous=None, **transform_kwargs):
        """Fit and transform the arrays

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates.

        **transform_kwargs : keyword args
            Keyword arguments required by the transform function.
        """
        self.fit(y, exogenous)
        return self.transform(y, exogenous, **transform_kwargs)

    @abc.abstractmethod
    def fit(self, y, exogenous):
        """Fit the transformer

        The purpose of the ``fit`` method is to learn a set of statistics or
        characteristics from the training set, and store them as "fit
        attributes" within the instance. A transformer *must* be fit before
        the transformation can be applied to a dataset in the ``transform``
        method.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features)
            The exogenous array of additional covariates.

        Returns
        -------
        self : BaseTransformer
            The scikit-learn convention is for the ``fit`` method to return
            the instance of the transformer, ``self``. This allows us to
            string ``fit(...).transform(...)`` calls together.
        """

    @abc.abstractmethod
    def transform(self, y, exogenous, **kwargs):
        """Transform the new array

        Apply the transformation to the array after learning the training set's
        characteristics in the ``fit`` method.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features)
            The exogenous array of additional covariates.

        **kwargs : keyword args
            Keyword arguments required by the transform function.

        Returns
        -------
        y : array-like or None
            The transformed y array

        exogenous : array-like or None
            The transformed exogenous array
        """


class UpdatableMixin:
    """Transformers that may update their params, like ARIMAs"""

    def _check_endog(self, y):
        if y is None:
            raise ValueError("endog array cannot be None when updating")

    def update_and_transform(self, y, exogenous, **kwargs):
        """Update the params and return the transformed arrays

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features)
            The exogenous array of additional covariates.

        **kwargs : keyword args
            Keyword arguments required by the transform function.
        """
