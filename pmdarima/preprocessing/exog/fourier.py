# -*- coding: utf-8 -*-

import numpy as np

from sklearn.utils.validation import check_is_fitted

from .base import BaseExogFeaturizer
from ..base import UpdatableMixin
from ._fourier import C_fourier_terms

__all__ = ['FourierFeaturizer']

sinpi = (lambda x: np.sin(np.pi * x))
cospi = (lambda x: np.cos(np.pi * x))


# Candidate for cythonization?
def _fourier_terms(p, times):
    # X = []
    # for e in p:
    #     X.append(sinpi(2 * e * times))
    #     X.append(cospi(2 * e * times))
    X = C_fourier_terms(p, times)
    return np.asarray(X).T


class FourierFeaturizer(BaseExogFeaturizer, UpdatableMixin):
    """Fourier terms for modeling seasonality

    This transformer creates an exogenous matrix containing terms from a
    Fourier series, up to order ``k``. It is based on ``R::forecast code`` [1].
    In practice, it permits us to fit a seasonal time series *without* seasonal
    order (i.e., ``seasonal=False``) by supplying decomposed seasonal Fourier
    terms as an exogenous array.

    The advantages of this technique, per Hyndman [2]:

        * It allows any length seasonality
        * The seasonal pattern is smooth for small values of K (but more wiggly
          seasonality can be handled by increasing K)
        * The short-term dynamics are easily handled with a simple ARMA error

    The disadvantage is that the seasonal periodicity of the time series is
    assumed to be fixed.

    Functionally, this is a featurizer. This means that exogenous features are
    *derived* from ``y``, as opposed to transforming an existing exog array.
    It also behaves slightly differently in the :func:`transform` stage than
    most other exogenous transformers in that ``exog`` is not a required arg,
    and it takes ``**kwargs``. See the :func:`transform` docstr for more info.

    Parameters
    ----------
    m : int
        The seasonal periodicity of the endogenous vector, y.

    k : int, optional (default=None)
        The number of sine and cosine terms (each) to include. I.e., if ``k``
        is 2, 4 new features will be generated. ``k`` must not exceed ``m/2``,
        which is the default value if not set. The value of ``k`` can be
        selected by minimizing the AIC.

    Notes
    -----
    * Helpful for long seasonal periods (large ``m``) where ``seasonal=True``
      seems to take a very long time to fit a model.

    References
    ----------
    .. [1] https://github.com/robjhyndman/forecast/blob/master/R/season.R
    .. [2] https://robjhyndman.com/hyndsight/longseasonality/
    """

    def __init__(self, m, k=None):
        self.m = m
        self.k = k

    def fit(self, y, exogenous=None):
        """Fit the transformer

        Computes the periods of all the Fourier terms. The values of ``y`` are
        not actually used; only the periodicity is used when computing Fourier
        terms.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. If specified, the
            Fourier terms will be column-bound on the right side of the matrix.
            Otherwise, the Fourier terms will be returned as the new exogenous
            array.
        """
        # Since we don't fit any params here, we can just check the params
        _, _ = self._check_y_exog(y, exogenous, null_allowed=True)

        m = self.m
        k = self.k
        if k is None:
            k = m // 2
        if 2 * k > m or k < 1:
            raise ValueError("k must be a positive integer not greater "
                             "than m//2")

        # Compute the periods of all Fourier terms. Since R allows multiple
        # seasonality and we do not, we can do this much more simply.
        p = ((np.arange(k) + 1) / m).astype(np.float64)  # 1:K / m

        # If sinpi is 0... maybe blow up?
        # if abs(2 * p - round(2 * p)) < np.finfo(y.dtype).eps:  # min eps

        self.p_ = p
        self.k_ = k
        self.n_ = y.shape[0]

        return self

    def transform(self, y, exogenous=None, n_periods=0, **_):
        """Create Fourier term features

        When an ARIMA is fit with an exogenous array, it must be forecasted
        with one also. Since at ``predict`` time in a pipeline we won't have
        ``y`` (and we may not yet have an ``exog`` array), we have to know how
        far into the future for which to compute Fourier terms (hence
        ``n_periods``).

        This method will compute the Fourier features for a given frequency and
        ``k`` term. Note that the ``y`` values are not used to compute these,
        so this does not pose a risk of data leakage.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array. This is unused and technically
            optional for the Fourier terms, since it uses the pre-computed
            ``n`` to calculate the seasonal Fourier terms.

        exogenous : array-like or None, shape=(n_samples, n_features), optional
            The exogenous array of additional covariates. If specified, the
            Fourier terms will be column-bound on the right side of the matrix.
            Otherwise, the Fourier terms will be returned as the new exogenous
            array.

        n_periods : int, optional (default=0)
            The number of periods in the future to forecast. If ``n_periods``
            is 0, will compute the Fourier features for the training set.
            ``n_periods`` corresponds to the number of samples that will be
            returned.
        """
        check_is_fitted(self, "p_")
        _, exog = self._check_y_exog(y, exogenous, null_allowed=True)

        if n_periods and exog is not None:
            if n_periods != exog.shape[0]:
                raise ValueError("If n_periods and exog are specified, "
                                 "n_periods must match dims of exogenous")

        times = np.arange(self.n_ + n_periods, dtype=np.float64) + 1
        X_fourier = _fourier_terms(self.p_, times)

        # Maybe trim if we're in predict mode... in that case, we only keep the
        # last n_periods rows in the matrix we've created
        if n_periods:
            X_fourier = X_fourier[-n_periods:, :]

        if exog is None:
            exog = X_fourier
        else:
            exog = np.hstack([exog, X_fourier])

        return y, exog

    def update_and_transform(self, y, exogenous, **kwargs):
        """Update the params and return the transformed arrays

        Since no parameters really get updated in the Fourier featurizer, all
        we do is compose forecasts for ``n_periods=len(y)`` and then update
        ``n_``.

        Parameters
        ----------
        y : array-like or None, shape=(n_samples,)
            The endogenous (time-series) array.

        exogenous : array-like or None, shape=(n_samples, n_features)
            The exogenous array of additional covariates.

        **kwargs : keyword args
            Keyword arguments required by the transform function.
        """
        check_is_fitted(self, "p_")

        self._check_endog(y)
        _, Xt = self.transform(y, exogenous, n_periods=len(y), **kwargs)

        # Update this *after* getting the exog features
        self.n_ += len(y)
        return y, Xt
