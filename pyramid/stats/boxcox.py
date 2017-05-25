# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Transform/inverse transform box-cox function

from __future__ import print_function, absolute_import, division
from sklearn.utils.validation import column_or_1d, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import boxcox
import numpy as np

__all__ = [
    'BoxCoxTransformer'
]


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """The Box-Cox transformation coerces a vector into more of a
    Gaussian distribution. This uses ``scipy.stats.boxcox`` under the
    hood.

    Parameters
    ----------
    lam : float, optional (default=None)
        The fit parameter. If ``lam`` is pre-defined, the fit method
        will simply return itself. Otherwise, ``scipy`` will optimize
        the ``lam`` parameter using MLE.

    alpha : float, optional (default=None)
        If alpha is not None, return the ``100 * (1-alpha)%`` confidence
        interval for ``lmbda`` as the third output argument. Must be between
        0.0 and 1.0.
    """
    def __init__(self, lam=None, alpha=None):
        self.lam = lam
        self.alpha = alpha

    def fit(self, y):
        self.fit_transform(y)
        return self

    def fit_transform(self, y, **other_args):
        # if lam is not yet fit, do the fit...
        if self.lam is None:
            y = column_or_1d(y)
            out = boxcox(y, lmbda=None, alpha=self.alpha)

            # depending on what alpha was, we might have len 2 or len 3 output. Rather
            # than have to test to unpack different lengths, just assign what we KNOW
            # will be present.
            self.lam_ = out[1]

            # if the conf interval is present...
            if len(out) > 2:
                self.confidence_ = out[2]
            return out[0]
        else:
            self.lam_ = self.lam

        # otherwise it's already fit.
        return self.transform(y)

    def transform(self, y):
        check_is_fitted(self, 'lam_')
        y = column_or_1d(y)
        return boxcox(y, lmbda=self.lam_, alpha=None)

    def inverse_transform(self, y):
        check_is_fitted(self, 'lam_')
        y = column_or_1d(y)
        lam = self.lam_

        # if it's zero
        np.power((y * lam) + 1, 1 / lam) - 1
