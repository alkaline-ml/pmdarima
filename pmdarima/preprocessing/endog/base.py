# -*- coding: utf-8 -*-

from sklearn.externals import six

import abc

from ..base import BaseTransformer


class BaseEndogTransformer(six.with_metaclass(abc.ABCMeta, BaseTransformer)):
    """A base class for endogenous array transformers"""

    def _check_y_exog(self, y, exog):
        """Check the endog and exog arrays"""
        y, exog = super(BaseEndogTransformer, self)._check_y_exog(y, exog)
        if y is None:
            raise ValueError("y must be non-None for endogenous transformers")
        return y, exog

    @abc.abstractmethod
    def fit(self, y, exog=None):
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

        exog : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. Not used for
            endogenous transformers. Default is None, and non-None values will
            serve as pass-through arrays.

        Returns
        -------
        self : BaseTransformer
            The scikit-learn convention is for the ``fit`` method to return
            the instance of the transformer, ``self``. This allows us to
            string ``fit(...).transform(...)`` calls together.
        """

    @abc.abstractmethod
    def transform(self, y, exog=None):
        """Transform the new array

        Apply the transformation to the array after learning the training set's
        characteristics in the ``fit`` method.

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
        y : array-like or None
            The transformed y array

        exog : array-like or None
            The transformed exogenous array
        """

    @abc.abstractmethod
    def inverse_transform(self, y, exog=None):
        """Inverse transform a transformed array

        Inverse the transformation on the transformed array.

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
