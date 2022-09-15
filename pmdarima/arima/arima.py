# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# A user-friendly wrapper to the statsmodels ARIMA that matches the familiar
# sklearn interface.

import pandas as pd
import numpy as np

import numpy.polynomial.polynomial as np_polynomial
import numbers
import warnings
from scipy.stats import gaussian_kde, norm
from sklearn.utils.validation import check_array
from statsmodels import api as sm

from . import _validation as val
from ..base import BaseARIMA
from ..compat.numpy import DTYPE  # DTYPE for arrays
from ..compat.sklearn import (
    check_is_fitted, if_delegate_has_method, safe_indexing
)
from ..compat import statsmodels as sm_compat
from ..compat import matplotlib as mpl_compat
from ..utils import if_has_delegate, is_iterable, check_endog, check_exog
from ..utils.visualization import _get_plt
from ..utils.array import diff_inv, diff

# Get the version
import pmdarima

__all__ = [
    'ARIMA',
    'ARMAtoMA'
]


def ARMAtoMA(ar, ma, max_deg):
    r"""
    Convert ARMA coefficients to infinite MA coefficients.

    Compute coefficients of MA model equivalent to given ARMA model.
    MA coefficients are cut off at max_deg.
    The same function as ARMAtoMA() in stats library of R

    Parameters
    ----------
    ar : array-like, shape=(n_orders,)
        The array of AR coefficients.

    ma : array-like, shape=(n_orders,)
        The array of MA coefficients.

    max_deg : int
        Coefficients are computed up to the order of max_deg.

    Returns
    -------
    np.ndarray, shape=(max_deg,)
        Equivalent MA coefficients.

    Notes
    -----
    Here is the derivation. Suppose ARMA model is defined as
    .. math::
    x_t - ar_1*x_{t-1} - ar_2*x_{t-2} - ... - ar_p*x_{t-p}\\
        = e_t + ma_1*e_{t-1} + ma_2*e_{t-2} + ... + ma_q*e_{t-q}
    namely
    .. math::
    (1 - \sum_{i=1}^p[ar_i*B^i]) x_t = (1 + \sum_{i=1}^q[ma_i*B^i]) e_t
    where :math:`B` is a backward operator.

    Equivalent MA model is
    .. math::
        x_t = (1 - \sum_{i=1}^p[ar_i*B^i])^{-1}\\
        * (1 + \sum_{i=1}^q[ma_i*B^i]) e_t\\
        = (1 + \sum_{i=1}[ema_i*B^i]) e_t
    where :math:``ema_i`` is a coefficient of equivalent MA model.
    The :math:``ema_i`` satisfies
    .. math::
        (1 - \sum_{i=1}^p[ar_i*B^i]) * (1 + \sum_{i=1}[ema_i*B^i]) \\
        = 1 + \sum_{i=1}^q[ma_i*B^i]
    thus
    .. math::
        \sum_{i=1}[ema_i*B^i] = \sum_{i=1}^p[ar_i*B^i] \\
        + \sum_{i=1}^p[ar_i*B^i] * \sum_{j=1}[ema_j*B^j] \\
        + \Sum_{i=1}^q[ma_i*B^i]
    therefore
    .. math::
        ema_i = ar_i (but 0 if i>p) \\
        + \Sum_{j=1}^{min(i-1,p)}[ar_j*ema_{i-j}] + ma_i(but 0 if i>q) \\
        = \sum_{j=1}{min(i,p)}[ar_j*ema_{i-j}(but 1 if j=i)] \\
        + ma_i(but 0 if i>q)

    Examples
    --------
    >>> ar = np.array([0.1])
    >>> ma = np.empty(0)
    >>> ARMAtoMA(ar, ma, 3)
    array[0.1, 0.01, 0.001]
    """
    p = len(ar)
    q = len(ma)
    ema = np.empty(max_deg)
    for i in range(0, max_deg):
        temp = ma[i] if i < q else 0.0
        for j in range(0, min(i + 1, p)):
            temp += ar[j] * (ema[i - j - 1] if i - j - 1 >= 0 else 1.0)
        ema[i] = temp
    return ema


def _aicc(model_results, nobs, add_constant):
    """Compute the corrected Akaike Information Criterion"""
    aic = model_results.aic

    # SARIMAX counts the constant in df_model if there was an intercept, so
    # only add one if no constant was included in the model
    df_model = model_results.df_model
    if add_constant:
        add_constant += 1  # add one for constant term
    return aic + 2. * df_model * (nobs / (nobs - df_model - 1.) - 1.)


def _append_to_endog(endog, new_y):
    """Append to the endogenous array

    Parameters
    ----------
    endog : np.ndarray, shape=(n_samples, [1])
        The existing endogenous array

    new_y : np.ndarray, shape=(n_samples)
        The new endogenous array to append
    """
    return np.concatenate((endog, new_y)) if \
        endog.ndim == 1 else \
        np.concatenate((endog.ravel(), new_y))[:, np.newaxis]


def _seasonal_prediction_with_confidence(
    arima_res,
    start,
    end,
    X,
    alpha,
    **kwargs,
):
    """Compute the prediction for a SARIMAX and get a conf interval

    Unfortunately, SARIMAX does not really provide a nice way to get the
    confidence intervals out of the box, so we have to perform the
    ``get_prediction`` code here and unpack the confidence intervals manually.

    Notes
    -----
    For internal use only.
    """
    results = arima_res.get_prediction(
        start=start,
        end=end,
        exog=X,
        **kwargs)

    f = results.predicted_mean
    conf_int = results.conf_int(alpha=alpha)
    if arima_res.specification['simple_differencing']:
        # If simple_differencing == True, statsmodels.get_prediction returns
        # mid and confidence intervals on differenced time series.
        # We have to invert differencing the mid and confidence intervals
        y_org = arima_res.model.orig_endog
        d = arima_res.model.orig_k_diff
        D = arima_res.model.orig_k_seasonal_diff
        period = arima_res.model.seasonal_periods
        # Forecast mid: undifferencing non-seasonal part
        if d > 0:
            y_sdiff = y_org if D == 0 else diff(y_org, period, D)
            f_temp = np.append(y_sdiff[-d:], f)
            f_temp = diff_inv(f_temp, 1, d)
            f = f_temp[(2 * d):]
        # Forecast mid: undifferencing seasonal part
        if D > 0 and period > 1:
            f_temp = np.append(y_org[-(D * period):], f)
            f_temp = diff_inv(f_temp, period, D)
            f = f_temp[(2 * D * period):]
        # confidence interval
        ar_poly = arima_res.polynomial_reduced_ar
        poly_diff = np_polynomial.polypow(np.array([1., -1.]), d)
        sdiff = np.zeros(period + 1)
        sdiff[0] = 1.
        sdiff[-1] = 1.
        poly_sdiff = np_polynomial.polypow(sdiff, D)
        ar = -np.polymul(ar_poly, np.polymul(poly_diff, poly_sdiff))[1:]
        ma = arima_res.polynomial_reduced_ma[1:]
        n_predMinus1 = end - start
        ema = ARMAtoMA(ar, ma, n_predMinus1)
        sigma2 = arima_res._params_variance[0]
        var = np.cumsum(np.append(1., ema * ema)) * sigma2
        q = results.dist.ppf(1. - alpha / 2, *results.dist_args)
        conf_int[:, 0] = f - q * np.sqrt(var)
        conf_int[:, 1] = f + q * np.sqrt(var)

    y_pred = check_endog(f, dtype=None, copy=False, preserve_series=True)
    conf_int = check_array(conf_int, copy=False, dtype=None)

    return y_pred, conf_int


class ARIMA(BaseARIMA):
    """An ARIMA estimator.

    An ARIMA, or autoregressive integrated moving average, is a
    generalization of an autoregressive moving average (ARMA) and is fitted to
    time-series data in an effort to forecast future points. ARIMA models can
    be especially efficacious in cases where data shows evidence of
    non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of interest is
    regressed on its own lagged (i.e., prior observed) values. The "MA" part
    indicates that the regression error is actually a linear combination of
    error terms whose values occurred contemporaneously and at various times
    in the past. The "I" (for "integrated") indicates that the data values
    have been replaced with the difference between their values and the
    previous values (and this differencing process may have been performed
    more than once). The purpose of each of these features is to make the model
    fit the data as well as possible.

    Non-seasonal ARIMA models are generally denoted ``ARIMA(p,d,q)`` where
    parameters ``p``, ``d``, and ``q`` are non-negative integers, ``p`` is the
    order (number of time lags) of the autoregressive model, ``d`` is the
    degree of differencing (the number of times the data have had past values
    subtracted), and ``q`` is the order of the moving-average model. Seasonal
    ARIMA models are usually denoted ``ARIMA(p,d,q)(P,D,Q)m``, where ``m``
    refers to the number of periods in each season, and the uppercase ``P``,
    ``D``, ``Q`` refer to the autoregressive, differencing, and moving average
    terms for the seasonal part of the ARIMA model.

    When two out of the three terms are zeros, the model may be referred to
    based on the non-zero parameter, dropping "AR", "I" or "MA" from the
    acronym describing the model. For example, ``ARIMA(1,0,0)`` is ``AR(1)``,
    ``ARIMA(0,1,0)`` is ``I(1)``, and ``ARIMA(0,0,1)`` is ``MA(1)``. [1]

    See notes for more practical information on the ``ARIMA`` class.

    Parameters
    ----------
    order : iterable or array-like, shape=(3,)
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters to use. ``p`` is the order (number of
        time lags) of the auto-regressive model, and is a non-negative integer.
        ``d`` is the degree of differencing (the number of times the data have
        had past values subtracted), and is a non-negative integer. ``q`` is
        the order of the moving-average model, and is a non-negative integer.

    seasonal_order : array-like, shape=(4,), optional (default=(0, 0, 0, 0))
        The (P,D,Q,s) order of the seasonal component of the model for the
        AR parameters, differences, MA parameters, and periodicity. ``D`` must
        be an integer indicating the integration order of the process, while
        ``P`` and ``Q`` may either be an integers indicating the AR and MA
        orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. ``S`` is an
        integer giving the periodicity (number of periods in season), often it
        is 4 for quarterly data or 12 for monthly data. Default is no seasonal
        effect.

    start_params : array-like, optional (default=None)
        Starting parameters for ``ARMA(p,q)``.  If None, the default is given
        by ``ARMA._fit_start_params``.

    method : str, optional (default='lbfgs')
        The ``method`` determines which solver from ``scipy.optimize``
        is used, and it can be chosen from among the following strings:

        - 'newton' for Newton-Raphson
        - 'nm' for Nelder-Mead
        - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
        - 'lbfgs' for limited-memory BFGS with optional box constraints
        - 'powell' for modified Powell's method
        - 'cg' for conjugate gradient
        - 'ncg' for Newton-conjugate gradient
        - 'basinhopping' for global basin-hopping solver

        The explicit arguments in ``fit`` are passed to the solver,
        with the exception of the basin-hopping solver. Each
        solver has several optional arguments that are not the same across
        solvers. These can be passed as **fit_kwargs

    maxiter : int, optional (default=50)
        The maximum number of function evaluations. Default is 50

    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If
        ``suppress_warnings`` is True, all of these warnings will be squelched.

    out_of_sample_size : int, optional (default=0)
        The number of examples from the tail of the time series to hold out
        and use as validation examples. The model will not be fit on these
        samples, but the observations will be added into the model's ``endog``
        and ``exog`` arrays so that future forecast values originate from the
        end of the endogenous vector. See :func:`update`.

        For instance::

            y = [0, 1, 2, 3, 4, 5, 6]
            out_of_sample_size = 2

            > Fit on: [0, 1, 2, 3, 4]
            > Score on: [5, 6]
            > Append [5, 6] to end of self.arima_res_.data.endog values

    scoring : str or callable, optional (default='mse')
        If performing validation (i.e., if ``out_of_sample_size`` > 0), the
        metric to use for scoring the out-of-sample data:

            * If a string, must be a valid metric name importable from
              ``sklearn.metrics``.
            * If a callable, must adhere to the function signature::

                def foo_loss(y_true, y_pred)

        Note that models are selected by *minimizing* loss. If using a
        maximizing metric (such as ``sklearn.metrics.r2_score``), it is the
        user's responsibility to wrap the function such that it returns a
        negative value for minimizing.

    scoring_args : dict, optional (default=None)
        A dictionary of key-word arguments to be passed to the
        ``scoring`` metric.

    trend : str or None, optional (default=None)
        The trend parameter. If ``with_intercept`` is True, ``trend`` will be
        used. If ``with_intercept`` is False, the trend will be set to a no-
        intercept value. If None and ``with_intercept``, 'c' will be used as
        a default.

    with_intercept : bool, optional (default=True)
        Whether to include an intercept term. Default is True.

    **sarimax_kwargs : keyword args, optional
        Optional arguments to pass to the SARIMAX constructor.
        Examples of potentially valuable kwargs:

          - time_varying_regression : boolean
            Whether or not coefficients on the exogenous regressors are allowed
            to vary over time.

          - enforce_stationarity : boolean
            Whether or not to transform the AR parameters to enforce
            stationarity in the auto-regressive component of the model.

          - enforce_invertibility : boolean
            Whether or not to transform the MA parameters to enforce
            invertibility in the moving average component of the model.

          - simple_differencing : boolean
            Whether or not to use partially conditional maximum likelihood
            estimation for seasonal ARIMA models. If True, differencing is
            performed prior to estimation, which discards the first
            :math:`s D + d` initial rows but results in a smaller
            state-space formulation. If False, the full SARIMAX model is
            put in state-space form so that all datapoints can be used in
            estimation. Default is False.

          - measurement_error: boolean
            Whether or not to assume the endogenous observations endog were
            measured with error. Default is False.

          - mle_regression : boolean
            Whether or not to use estimate the regression coefficients for the
            exogenous variables as part of maximum likelihood estimation or
            through the Kalman filter (i.e. recursive least squares). If
            time_varying_regression is True, this must be set to False.
            Default is True.

          - hamilton_representation : boolean
            Whether or not to use the Hamilton representation of an ARMA
            process (if True) or the Harvey representation (if False).
            Default is False.

          - concentrate_scale : boolean
            Whether or not to concentrate the scale (variance of the error
            term) out of the likelihood. This reduces the number of parameters
            estimated by maximum likelihood by one, but standard errors will
            then not be available for the scale parameter.

    Attributes
    ----------
    arima_res_ : ModelResultsWrapper
        The model results, per statsmodels

    endog_index_ : pd.Series or None
        If the fitted endog array is a ``pd.Series``, this value will be
        non-None and is used to validate args for in-sample predictions with
        non-integer start/end indices

    oob_ : float
        The MAE or MSE of the out-of-sample records, if ``out_of_sample_size``
        is > 0, else np.nan

    oob_preds_ : np.ndarray or None
        The predictions for the out-of-sample records, if
        ``out_of_sample_size`` is > 0, else None

    Notes
    -----
    * The model internally wraps the statsmodels `SARIMAX class <https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>`_
    * After the model fit, many more methods will become available to the
      fitted model (i.e., :func:`pvalues`, :func:`params`, etc.). These are
      delegate methods which wrap the internal ARIMA results instance.

    See Also
    --------
    :func:`pmdarima.arima.auto_arima`

    References
    ----------
    .. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average
    """  # noqa: E501
    def __init__(
        self,
        order,
        seasonal_order=(0, 0, 0, 0),
        start_params=None,
        method='lbfgs',
        maxiter=50,
        suppress_warnings=False,
        out_of_sample_size=0,
        scoring='mse',
        scoring_args=None,
        trend=None,
        with_intercept=True,
        **sarimax_kwargs,
    ):

        # Future proofing: this isn't currently required--sklearn doesn't
        # need a super call
        super(ARIMA, self).__init__()

        self.order = order
        self.seasonal_order = seasonal_order
        self.start_params = start_params
        self.method = method
        self.maxiter = maxiter
        self.suppress_warnings = suppress_warnings
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = scoring_args
        self.trend = trend
        self.with_intercept = with_intercept

        # TODO: Remove these warnings in a later version
        for deprecated_key, still_in_use in (
                ('disp', True),
                ('callback', True),
                ('transparams', False),
                ('solver', False)):
            if sarimax_kwargs.pop(deprecated_key, None):
                msg = ("'%s' is deprecated in the ARIMA constructor and will "
                       "be removed in a future release." % deprecated_key)
                if still_in_use:
                    msg += " Pass via **fit_kwargs instead"
                warnings.warn(msg, DeprecationWarning)

        self.sarimax_kwargs = sarimax_kwargs

    def _fit(self, y, X=None, **fit_args):
        """Internal fit"""

        # This wrapper is used for fitting either an ARIMA or a SARIMAX
        def _fit_wrapper():
            method = self.method
            trend = self.trend

            if method is None:
                raise ValueError("Expected non-None value for `method`")

            # this considers `with_intercept` truthy, so if auto_arima gets
            # here without explicitly changing `with_intercept` from 'auto' we
            # will treat it as True.
            if trend is None and self.with_intercept:
                trend = 'c'

            # create the SARIMAX
            sarimax_kwargs = \
                {} if not self.sarimax_kwargs else self.sarimax_kwargs
            seasonal_order = \
                sm_compat.check_seasonal_order(self.seasonal_order)
            arima = sm.tsa.statespace.SARIMAX(
                endog=y, exog=X, order=self.order,
                seasonal_order=seasonal_order,
                trend=trend,
                **sarimax_kwargs)

            # actually fit the model, now. If this was called from 'update',
            # give priority to start_params from the fit_args
            start_params = fit_args.pop("start_params", self.start_params)

            # Same for 'maxiter' if called from update. Also allows it to be
            # passed as a fit arg, if a user does it explicitly.
            _maxiter = self.maxiter
            if _maxiter is None:
                raise ValueError("Expected non-None value for `maxiter`")

            # If maxiter is provided in the fit_args by a savvy user or the
            # update method, we should default to their preference
            _maxiter = fit_args.pop("maxiter", _maxiter)

            disp = fit_args.pop("disp", 0)
            fitted = arima.fit(
                start_params=start_params,
                method=method,
                maxiter=_maxiter,
                disp=disp,
                **fit_args,
            )

            return arima, fitted

        # sometimes too many warnings...
        if self.suppress_warnings:
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                fit, self.arima_res_ = _fit_wrapper()
        else:
            fit, self.arima_res_ = _fit_wrapper()

        # Set df_model attribute for SARIMAXResults object
        sm_compat.bind_df_model(fit, self.arima_res_)

        # if the model is fit with an X array, it must
        # be predicted with one as well.
        self.fit_with_exog_ = X is not None

        # Save nobs since we might change it later if using OOB
        self.nobs_ = y.shape[0]

        # As of version 0.7.2, start saving the version with the model so
        # we can track changes over time.
        self.pkg_version_ = pmdarima.__version__
        return self

    def fit(self, y, X=None, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations with an
        optional matrix of ``X`` variables.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series to which to fit the ``ARIMA`` estimator. This may
            either be a Pandas ``Series`` object (statsmodels can internally
            use the dates in the index), or a numpy array. This should be a
            one-dimensional array of floats, and should not contain any
            ``np.nan`` or ``np.inf`` values.

        X : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        **fit_args : dict or kwargs
            Any keyword arguments to pass to the statsmodels ARIMA fit.
        """
        y = check_endog(y, dtype=DTYPE, preserve_series=True)
        n_samples = y.shape[0]

        # See issue 499
        self.endog_index_ = y.index if isinstance(y, pd.Series) else None

        # if exog was included, check the array...
        if X is not None:
            X = check_exog(X, force_all_finite=False, copy=False, dtype=DTYPE)

        # determine the CV args, if any
        cv = self.out_of_sample_size
        scoring = val.get_scoring_metric(self.scoring)

        # don't allow negative, don't allow > n_samples
        cv = max(cv, 0)

        # if cv is too big, raise
        if cv >= n_samples:
            raise ValueError("out-of-sample size must be less than number "
                             "of samples!")

        # If we want to get a score on the out-of-sample, we need to trim
        # down the size of our y vec for fitting. Addressed due to Issue #28
        cv_samples = None
        cv_exog = None
        if cv:
            cv_samples = y[-cv:]
            y = y[:-cv]

            # This also means we have to address X
            if X is not None:
                n_exog = X.shape[0]
                cv_exog = safe_indexing(X, slice(n_exog - cv, n_exog))
                X = safe_indexing(X, slice(0, n_exog - cv))

        # Internal call
        self._fit(y, X, **fit_args)

        # now make a forecast if we're validating to compute the
        # out-of-sample score
        if cv_samples is not None:
            # get the predictions (use self.predict, which calls forecast
            # from statsmodels internally)
            pred = self.predict(n_periods=cv, X=cv_exog)
            scoring_args = {} if not self.scoring_args else self.scoring_args
            self.oob_ = scoring(cv_samples, pred, **scoring_args)
            self.oob_preds_ = pred

            # If we compute out of sample scores, we have to now update the
            # observed time points so future forecasts originate from the end
            # of our y vec
            self.update(cv_samples, cv_exog, **fit_args)
        else:
            self.oob_ = np.nan
            self.oob_preds_ = None

        return self

    def _check_exog(self, X):
        # if we fit with exog, make sure one was passed, or else fail out:
        if self.fit_with_exog_:
            if X is None:
                raise ValueError('When an ARIMA is fit with an X '
                                 'array, it must also be provided one for '
                                 'predicting or updating observations.')
            else:
                return check_exog(X, force_all_finite=True, dtype=DTYPE)
        return None

    def predict_in_sample(
        self,
        X=None,
        start=None,
        end=None,
        dynamic=False,
        return_conf_int=False,
        alpha=0.05,
        **kwargs,
    ):
        """Generate in-sample predictions from the fit ARIMA model.

        Predicts the original training (in-sample) time series values. This can
        be useful when wanting to visualize the fit, and qualitatively inspect
        the efficacy of the model, or when wanting to compute the residuals
        of the model.

        Parameters
        ----------
        X : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        start : int or object, optional (default=None)
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start. Note that if this value is less than
            ``d``, the order of differencing, an error will be raised.
            Alternatively, a non-int value can be given if the model was fit
            on a ``pd.Series`` with an object-type index, like a timestamp.

        end : int or object, optional (default=None)
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start. Alternatively, a non-int value can be
            given if the model was fit on a ``pd.Series`` with an object-type
            index, like a timestamp.

        dynamic : bool, optional (default=False)
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.

        return_conf_int : bool, optional (default=False)
            Whether to get the confidence intervals of the forecasts.

        alpha : float, optional (default=0.05)
            The confidence intervals for the forecasts are (1 - alpha) %

        Returns
        -------
        preds : array
            The predicted values.

        conf_int : array-like, shape=(n_periods, 2), optional
            The confidence intervals for the predictions. Only returned if
            ``return_conf_int`` is True.
        """
        check_is_fitted(self, 'arima_res_')

        # issue #499: support prediction with non-integer start/end indices
        # issue #286: we can't produce valid preds for start < diff value
        # we can only do the validation for issue 286 if `start` is an int
        d = self.order[1]
        if isinstance(start, numbers.Integral) and start < d:
            raise ValueError(
                f"In-sample predictions undefined for start={start} when d={d}"
            )

        # if we fit with exog, make sure one was passed:
        X = self._check_exog(X)  # type: np.ndarray
        results_wrapper = self.arima_res_

        # If not returning the confidence intervals, we have it really easy
        if not return_conf_int:
            preds = results_wrapper.predict(
                exog=X,
                start=start,
                end=end,
                dynamic=dynamic,
            )

            return preds

        # We cannot get confidence intervals if dynamic is true
        if dynamic:
            warnings.warn("Cannot produce in-sample confidence intervals for "
                          "dynamic=True. Setting dynamic=False")
            dynamic = False

        # Otherwise confidence intervals are requested...
        preds, conf_int = _seasonal_prediction_with_confidence(
            arima_res=results_wrapper,
            start=start,
            end=end,
            X=X,
            alpha=alpha,
            dynamic=dynamic,
        )

        return preds, conf_int

    def predict(self,
                n_periods=10,
                X=None,
                return_conf_int=False,
                alpha=0.05,
                **kwargs):  # TODO: remove kwargs after exog disappears
        """Forecast future values

        Generate predictions (forecasts) ``n_periods`` in the future.
        Note that if ``exogenous`` variables were used in the model fit, they
        will be expected for the predict procedure and will fail otherwise.

        Parameters
        ----------
        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        X : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        return_conf_int : bool, optional (default=False)
            Whether to get the confidence intervals of the forecasts.

        alpha : float, optional (default=0.05)
            The confidence intervals for the forecasts are (1 - alpha) %

        Returns
        -------
        forecasts : array-like, shape=(n_periods,)
            The array of fore-casted values.

        conf_int : array-like, shape=(n_periods, 2), optional
            The confidence intervals for the forecasts. Only returned if
            ``return_conf_int`` is True.
        """
        check_is_fitted(self, 'arima_res_')
        if not isinstance(n_periods, int):
            raise TypeError("n_periods must be an int")

        # if we fit with exog, make sure one was passed:
        X = self._check_exog(X)  # type: np.ndarray
        if X is not None and X.shape[0] != n_periods:
            raise ValueError(
                f'X array dims (n_rows) != n_periods. Received '
                f'n_rows={X.shape[0]} and n_periods={n_periods}'
            )

        # f = self.arima_res_.forecast(steps=n_periods, exog=X)
        arima = self.arima_res_
        end = arima.nobs + n_periods - 1

        f, conf_int = _seasonal_prediction_with_confidence(
            arima_res=arima,
            start=arima.nobs,
            end=end,
            X=X,
            alpha=alpha)

        if return_conf_int:
            # The confidence intervals may be a Pandas frame if it comes from
            # SARIMAX & we want Numpy. We will to duck type it so we don't add
            # new explicit requirements for the package
            return f, check_array(conf_int, force_all_finite=False)
        return f

    def __getstate__(self):
        """I am being pickled..."""

        # In versions <0.9.0, if this already contains a pointer to a
        # "saved state" model, we deleted that model and replaced it with the
        # new one.
        # In version >= v0.9.0, we keep the old model around, since that's how
        # the user expects it should probably work (otherwise unpickling the
        # previous state of the model would raise an OSError).
        # In version >= 1.1.0, we allow statsmodels results wrappers to be
        # bundled into the same pickle file (see Issue #48) which is possible
        # due to statsmodels v0.9.0+. As a result, we no longer really need
        # this subhook...
        return self.__dict__

    def __setstate__(self, state):
        """I am being unpickled..."""
        self.__dict__ = state

        # Warn for unpickling a different version's model
        self._warn_for_older_version()
        return self

    def _warn_for_older_version(self):
        # Added in v0.8.1 - check for the version pickled under and warn
        # if it's different from the current version
        do_warn = False
        modl_version = None
        this_version = pmdarima.__version__

        try:
            modl_version = getattr(self, 'pkg_version_')

            # Either < or > or '-dev' vs. release version
            if modl_version != this_version:
                do_warn = True
        except AttributeError:
            # Either wasn't fit when pickled or didn't have the attr due to
            # it being an older version. If it wasn't fit, it will be missing
            # the arima_res_ attr.
            if hasattr(self, 'arima_res_'):  # it was fit, but is older
                do_warn = True
                modl_version = '<0.8.1'

            # else: it may not have the model (not fit) and still be older,
            # but we can't detect that.

        # Means it was older
        if do_warn:
            warnings.warn("You've deserialized an ARIMA from a version (%s) "
                          "that does not match your installed version of "
                          "pmdarima (%s). This could cause unforeseen "
                          "behavior."
                          % (modl_version, this_version), UserWarning)

    def __str__(self):
        """Different from __repr__, returns a debug string used in logging"""
        p, d, q = self.order
        P, D, Q, m = self.seasonal_order
        int_str = "intercept"
        with_intercept = self.with_intercept
        return (
            " ARIMA({p},{d},{q})({P},{D},{Q})[{m}] {intercept}".format(
                p=p,
                d=d,
                q=q,
                P=P,
                D=D,
                Q=Q,
                m=m,
                # just for consistent spacing
                intercept=int_str if with_intercept else " " * len(int_str)
            )
        )

    def update(self, y, X=None, maxiter=None, **kwargs):
        """Update the model fit with additional observed endog/exog values.

        Updating an ARIMA adds new observations to the model, updating the
        MLE of the parameters accordingly by performing several new iterations
        (``maxiter``) from the existing model parameters.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series data to add to the endogenous samples on which the
            ``ARIMA`` estimator was previously fit. This may either be a Pandas
            ``Series`` object or a numpy array. This should be a one-
            dimensional array of finite floats.

        X : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If the model was
            fit with an exogenous array of covariates, it will be required for
            updating the observed values.

        maxiter : int, optional (default=None)
            The number of iterations to perform when updating the model. If
            None, will perform ``max(5, n_samples // 10)`` iterations.

        **kwargs : keyword args
            Any keyword args that should be passed as ``**fit_kwargs`` in the
            new model fit.

        Notes
        -----
        * Internally, this calls ``fit`` again using the OLD model parameters
          as the starting parameters for the new model's MLE computation.
        """
        check_is_fitted(self, 'arima_res_')
        model_res = self.arima_res_

        # Allow updating with a scalar if the user is just adding a single
        # sample.
        if not is_iterable(y):
            y = [y]

        # validate the new samples to add
        y = check_endog(y, dtype=DTYPE, preserve_series=True)
        n_samples = y.shape[0]

        # if X is None and new X provided, or vice versa, raise
        X = self._check_exog(X)  # type: np.ndarray

        # ensure the k_exog matches
        if X is not None:
            k_exog = model_res.model.k_exog
            n_exog, exog_dim = X.shape

            if X.shape[1] != k_exog:
                raise ValueError(
                    f"Dim mismatch in fit `X` ({k_exog}) and new "
                    f"`X` ({exog_dim})"
                )

            # make sure the number of samples in X match the number
            # of samples in the endog
            if n_exog != n_samples:
                raise ValueError(
                    f"Dim mismatch in n_samples (y={n_samples}, X={n_exog})"
                )

        # first concatenate the original data (might be 2d or 1d)
        y = np.squeeze(_append_to_endog(model_res.data.endog, y))

        # Now create the new X.
        if X is not None:
            # Concatenate
            X_prime = np.concatenate((model_res.data.exog, X), axis=0)
        else:
            # Just so it's in the namespace
            X_prime = None

        # This is currently arbitrary... but it's here to avoid accidentally
        # overfitting a user's model. Would be nice to find some papers that
        # describe the best way to set this.
        if maxiter is None:
            maxiter = max(5, n_samples // 10)

        # Get the model parameters, then we have to "fit" a new one. If you're
        # reading this source code, don't panic! We're not just fitting a new
        # arbitrary model. Statsmodels does not handle patching new samples in
        # very well, so we seed the new model with the existing parameters.
        params = model_res.params
        self._fit(y, X_prime, start_params=params, maxiter=maxiter, **kwargs)

        # Behaves like `fit`
        return self

    @if_delegate_has_method('arima_res_')
    def aic(self):
        """Get the AIC, the Akaike Information Criterion:

            :code:`-2 * llf + 2 * df_model`

        Where ``df_model`` (the number of degrees of freedom in the model)
        includes all AR parameters, MA parameters, constant terms parameters
        on constant terms and the variance.

        Returns
        -------
        aic : float
            The AIC

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Akaike_information_criterion
        """
        return self.arima_res_.aic

    # TODO: this looks like it's implemented on statsmodels' master
    @if_has_delegate('arima_res_')
    def aicc(self):
        """Get the AICc, the corrected Akaike Information Criterion:

            :code:`AIC + 2 * df_model * (df_model + 1) / (nobs - df_model - 1)`

        Where ``df_model`` (the number of degrees of freedom in the model)
        includes all AR parameters, MA parameters, constant terms parameters
        on constant terms and the variance. And ``nobs`` is the sample size.

        Returns
        -------
        aicc : float
            The AICc

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
        """
        # FIXME:
        # this code should really be added to statsmodels. Rewrite
        # this function to reflect other metric implementations if/when
        # statsmodels incorporates AICc
        return _aicc(self.arima_res_,
                     self.nobs_,
                     not self.with_intercept)

    @if_delegate_has_method('arima_res_')
    def arparams(self):
        """Get the parameters associated with the AR coefficients in the model.

        Returns
        -------
        arparams : array-like
            The AR coefficients.
        """
        return self.arima_res_.arparams

    @if_delegate_has_method('arima_res_')
    def arroots(self):
        """The roots of the AR coefficients are the solution to:

            :code:`(1 - arparams[0] * z - arparams[1] * z^2 - ... - arparams[
            p-1] * z^k_ar) = 0`

        Stability requires that the roots in modulus lie outside the unit
        circle.

        Returns
        -------
        arroots : array-like
            The roots of the AR coefficients.
        """
        return self.arima_res_.arroots

    @if_delegate_has_method('arima_res_')
    def bic(self):
        """Get the BIC, the Bayes Information Criterion:

            :code:`-2 * llf + log(nobs) * df_model`

        Where if the model is fit using conditional sum of squares, the
        number of observations ``nobs`` does not include the ``p`` pre-sample
        observations.

        Returns
        -------
        bse : float
            The BIC

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Bayesian_information_criterion
        """
        return self.arima_res_.bic

    @if_delegate_has_method('arima_res_')
    def bse(self):
        """Get the standard errors of the parameters. These are
        computed using the numerical Hessian.

        Returns
        -------
        bse : array-like
            The BSE
        """
        return self.arima_res_.bse

    @if_delegate_has_method('arima_res_')
    def conf_int(self, alpha=0.05, **kwargs):
        r"""Returns the confidence interval of the fitted parameters.

        Returns
        -------
        alpha : float, optional (default=0.05)
            The significance level for the confidence interval. ie.,
            the default alpha = .05 returns a 95% confidence interval.

        **kwargs : keyword args or dict
            Keyword arguments to pass to the confidence interval function.
            Could include 'cols' or 'method'
        """
        return self.arima_res_.conf_int(alpha=alpha, **kwargs)

    @if_delegate_has_method('arima_res_')
    def df_model(self):
        """The model degrees of freedom: ``k_exog`` + ``k_trend`` +
        ``k_ar`` + ``k_ma``.

        Returns
        -------
        df_model : array-like
            The degrees of freedom in the model.
        """
        return self.arima_res_.df_model

    @if_delegate_has_method('arima_res_')
    def df_resid(self):
        """Get the residual degrees of freedom:

            :code:`nobs - df_model`

        Returns
        -------
        df_resid : array-like
            The residual degrees of freedom.
        """
        return self.arima_res_.df_resid

    @if_delegate_has_method('arima_res_')
    def fittedvalues(self):
        """Get the fitted values from the model

        Returns
        -------
        fittedvalues : array-like
            The predicted values for the original series
        """
        return self.arima_res_.fittedvalues

    @if_delegate_has_method('arima_res_')
    def hqic(self):
        """Get the Hannan-Quinn Information Criterion:

            :code:`-2 * llf + 2 * (`df_model`) * log(log(nobs))`

        Like :func:`bic` if the model is fit using conditional sum of squares
        then the ``k_ar`` pre-sample observations are not counted in ``nobs``.

        Returns
        -------
        hqic : float
            The HQIC

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Hannan-Quinn_information_criterion
        """
        return self.arima_res_.hqic

    @if_delegate_has_method('arima_res_')
    def maparams(self):
        """Get the value of the moving average coefficients.

        Returns
        -------
        maparams : array-like
            The MA coefficients.
        """
        return self.arima_res_.maparams

    @if_delegate_has_method('arima_res_')
    def maroots(self):
        """The roots of the MA coefficients are the solution to:

            :code:`(1 + maparams[0] * z + maparams[1] * z^2 + ... + maparams[
            q-1] * z^q) = 0`

        Stability requires that the roots in modules lie outside the unit
        circle.

        Returns
        -------
        maroots : array-like
            The MA roots.
        """
        return self.arima_res_.maroots

    def oob(self):
        """If the model was built with ``out_of_sample_size`` > 0, a validation
        score will have been computed. Otherwise it will be np.nan.

        Returns
        -------
        oob_ : float
            The "out-of-bag" score.
        """
        return self.oob_

    @if_delegate_has_method('arima_res_')
    def params(self):
        """Get the parameters of the model. The order of variables is the trend
        coefficients and the :func:`k_exog` exogenous coefficients, then the
        :func:`k_ar` AR coefficients, and finally the :func:`k_ma` MA
        coefficients.

        Returns
        -------
        params : array-like
            The parameters of the model.
        """
        return self.arima_res_.params

    @if_delegate_has_method('arima_res_')
    def pvalues(self):
        """Get the p-values associated with the t-values of the coefficients.
        Note that the coefficients are assumed to have a Student's T
        distribution.

        Returns
        -------
        pvalues : array-like
            The p-values.
        """
        return self.arima_res_.pvalues

    @if_delegate_has_method('arima_res_')
    def resid(self):
        """Get the model residuals. If the model is fit using 'mle', then the
        residuals are created via the Kalman Filter. If the model is fit
        using 'css' then the residuals are obtained via
        ``scipy.signal.lfilter`` adjusted such that the first :func:`k_ma`
        residuals are zero. These zero residuals are not returned.

        Returns
        -------
        resid : array-like
            The model residuals.
        """
        return self.arima_res_.resid

    @if_delegate_has_method('arima_res_')
    def summary(self):
        """Get a summary of the ARIMA model"""
        return self.arima_res_.summary()

    @if_has_delegate('arima_res_')
    def to_dict(self):
        """Get the ARIMA model as a dictionary

        Return the dictionary representation of the ARIMA model

        Returns
        -------
        res : dictionary
            The ARIMA model as a dictionary.
        """
        seasonal = sm_compat.check_seasonal_order(self.seasonal_order)
        return {
            'pvalues': self.pvalues(),
            'resid': self.resid(),
            'order': self.order,
            'seasonal_order': seasonal,
            'oob': self.oob(),
            'aic': self.aic(),
            'aicc': self.aicc(),
            'bic': self.bic(),
            'bse': self.bse(),
            'params': self.params()
        }

    @if_has_delegate('arima_res_')
    def plot_diagnostics(self, variable=0, lags=10, fig=None, figsize=None):
        """Plot an ARIMA's diagnostics.

        Diagnostic plots for standardized residuals of one endogenous variable

        Parameters
        ----------
        variable : integer, optional
            Index of the endogenous variable for which the diagnostic plots
            should be created. Default is 0.

        lags : integer, optional
            Number of lags to include in the correlogram. Default is 10.

        fig : Matplotlib Figure instance, optional
            If given, subplots are created in this figure instead of in a new
            figure. Note that the 2x2 grid will be created in the provided
            figure using `fig.add_subplot()`.

        figsize : tuple, optional
            If a figure is created, this argument allows specifying a size.
            The tuple is (width, height).

        Notes
        -----
        Produces a 2x2 plot grid with the following plots (ordered clockwise
        from top left):

        1. Standardized residuals over time
        2. Histogram plus estimated density of standardized residuals, along
           with a Normal(0,1) density plotted for reference.
        3. Normal Q-Q plot, with Normal reference line.
        4. Correlogram

        See Also
        --------
        statsmodels.graphics.gofplots.qqplot
        pmdarima.utils.visualization.plot_acf

        References
        ----------
        .. [1] https://www.statsmodels.org/dev/_modules/statsmodels/tsa/statespace/mlemodel.html#MLEResults.plot_diagnostics
        """  # noqa: E501
        # implicitly checks whether installed, and does our backend magic:
        _get_plt()

        # We originally delegated down to SARIMAX model wrapper, but
        # statsmodels makes it difficult to trust their API, so we just re-
        # implemented a common method for all results wrappers.
        from statsmodels.graphics import utils as sm_graphics
        fig = sm_graphics.create_mpl_fig(fig, figsize)

        res_wpr = self.arima_res_
        data = res_wpr.data

        # Eliminate residuals associated with burned or diffuse likelihoods.
        # The statsmodels code for the Kalman Filter takes the loglik_burn
        # as a parameter:

        # loglikelihood_burn : int, optional
        #     The number of initial periods during which the loglikelihood is
        #     not recorded. Default is 0.

        # If the class has it, it's a SARIMAX and we'll use it. Otherwise we
        # will just access the residuals as we normally would...
        if hasattr(res_wpr, 'loglikelihood_burn'):
            # This is introduced in the bleeding edge version, but is not
            # backwards compatible with 0.9.0 and less:
            d = res_wpr.loglikelihood_burn
            if hasattr(res_wpr, 'nobs_diffuse'):
                d = np.maximum(d, res_wpr.nobs_diffuse)

            resid = res_wpr.filter_results\
                           .standardized_forecasts_error[variable, d:]
        else:
            # This gets the residuals, but they need to be standardized
            d = 0
            r = res_wpr.resid
            resid = (r - np.nanmean(r)) / np.nanstd(r)

        # Top-left: residuals vs time
        ax = fig.add_subplot(221)
        if hasattr(data, 'dates') and data.dates is not None:
            x = data.dates[d:]._mpl_repr()
        else:
            x = np.arange(len(resid))
        ax.plot(x, resid)
        ax.hlines(0, x[0], x[-1], alpha=0.5)
        ax.set_xlim(x[0], x[-1])
        ax.set_title('Standardized residual')

        # Top-right: histogram, Gaussian kernel density, Normal density
        # Can only do histogram and Gaussian kernel density on the non-null
        # elements
        resid_nonmissing = resid[~(np.isnan(resid))]
        ax = fig.add_subplot(222)
        # temporarily disable Deprecation warning, normed -> density
        # hist needs to use `density` in future when minimum matplotlib has it
        # 'normed' argument is no longer supported in matplotlib since
        # version 3.2.0. New function added for backwards compatibility
        with warnings.catch_warnings(record=True):
            ax.hist(
                resid_nonmissing,
                label='Hist',
                **mpl_compat.mpl_hist_arg()
            )

        kde = gaussian_kde(resid_nonmissing)
        xlim = (-1.96 * 2, 1.96 * 2)
        x = np.linspace(xlim[0], xlim[1])
        ax.plot(x, kde(x), label='KDE')
        ax.plot(x, norm.pdf(x), label='N(0,1)')
        ax.set_xlim(xlim)
        ax.legend()
        ax.set_title('Histogram plus estimated density')

        # Bottom-left: QQ plot
        ax = fig.add_subplot(223)
        from statsmodels.graphics import gofplots
        gofplots.qqplot(resid_nonmissing, line='s', ax=ax)
        ax.set_title('Normal Q-Q')

        # Bottom-right: Correlogram
        ax = fig.add_subplot(224)
        from statsmodels.graphics import tsaplots
        tsaplots.plot_acf(resid, ax=ax, lags=lags)
        ax.set_title('Correlogram')

        ax.set_ylim(-1, 1)

        return fig
