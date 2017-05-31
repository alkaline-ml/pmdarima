# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# A much more user-friendly wrapper to the statsmodels ARIMA.
# Mimics the familiar sklearn interface.

from __future__ import print_function, absolute_import, division

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.metaestimators import if_delegate_has_method
from statsmodels.tsa.arima_model import ARIMA as _ARIMA
import warnings

# The DTYPE we'll use for everything here. Since there are
# lots of spots where we define the DTYPE in a numpy array,
# it's easier to define as a global for this module.
import numpy as np
DTYPE = np.float64

__all__ = [
    'ARIMA'
]


class ARIMA(BaseEstimator):
    """An ARIMA, or autoregressive integrated moving average, is a generalization of an autoregressive
    moving average (ARMA) and is fitted to time-series data in an effort to forecast future points.
    ARIMA models can be especially efficacious in cases where data shows evidence of non-stationarity.

    The "AR" part of ARIMA indicates that the evolving variable of interest is regressed on its own
    lagged (i.e., prior observed) values. The "MA" part indicates that the regression error is actually a linear
    combination of error terms whose values occurred contemporaneously and at various times in the past.
    The "I" (for "integrated") indicates that the data values have been replaced with the difference
    between their values and the previous values (and this differencing process may have been performed
    more than once). The purpose of each of these features is to make the model fit the data as well as possible.

    Non-seasonal ARIMA models are generally denoted ``ARIMA(p,d,q)`` where parameters ``p``, ``d``, and ``q`` are
    non-negative integers, ``p`` is the order (number of time lags) of the autoregressive model, ``d`` is the degree
    of differencing (the number of times the data have had past values subtracted), and ``q`` is the order of the
    moving-average model. Seasonal ARIMA models are usually denoted ``ARIMA(p,d,q)(P,D,Q)m``, where ``m`` refers
    to the number of periods in each season, and the uppercase ``P``, ``D``, ``Q`` refer to the autoregressive,
    differencing, and moving average terms for the seasonal part of the ARIMA model.

    When two out of the three terms are zeros, the model may be referred to based on the non-zero parameter,
    dropping "AR", "I" or "MA" from the acronym describing the model. For example, ``ARIMA(1,0,0)`` is ``AR(1)``,
    ``ARIMA(0,1,0)`` is ``I(1)``, and ``ARIMA(0,0,1)`` is ``MA(1)``. [1]

    See notes for more practical information on the ``ARIMA`` class.


    Parameters
    ----------
    order : iterable or array-like, shape=(3,)
        The (p,d,q) order of the model for the number of AR parameters, differences, and MA parameters
        to use. ``p`` is the order (number of time lags) of the auto-regressive model, and is a non-
        negative integer. ``d`` is the degree of differencing (the number of times the data have
        had past values subtracted), and is a non-negative integer. ``q`` is the order of the moving-
        average model, and is a non-negative integer.

    start_params : array-like, optional (default=None)
        Starting parameters for ``ARMA(p,q)``.  If None, the default is given
        by ``ARMA._fit_start_params``.

    transparams : bool, optional (default=True)
        Whehter or not to transform the parameters to ensure stationarity.
        Uses the transformation suggested in Jones (1980).  If False,
        no checking for stationarity or invertibility is done.

    method : str, one of {'css-mle','mle','css'}, optional (default='css-mle')
        This is the loglikelihood to maximize.  If "css-mle", the
        conditional sum of squares likelihood is maximized and its values
        are used as starting values for the computation of the exact
        likelihood via the Kalman filter.  If "mle", the exact likelihood
        is maximized via the Kalman Filter.  If "css" the conditional sum
        of squares likelihood is maximized.  All three methods use
        `start_params` as starting parameters.  See above for more
        information.

    trend : str {'c','nc'}, optional (default='c')
        Whether to include a constant or not.  'c' includes constant,
        'nc' no constant.

    solver : str or None, optional (default='lbfgs')
        Solver to be used.  The default is 'lbfgs' (limited memory
        Broyden-Fletcher-Goldfarb-Shanno).  Other choices are 'bfgs',
        'newton' (Newton-Raphson), 'nm' (Nelder-Mead), 'cg' -
        (conjugate gradient), 'ncg' (non-conjugate gradient), and
        'powell'. By default, the limited memory BFGS uses m=12 to
        approximate the Hessian, projected gradient tolerance of 1e-8 and
        factr = 1e2. You can change these by using kwargs.

    maxiter : int, optional (default=50)
        The maximum number of function evaluations. Default is 50.

    disp : int, optional (default=0)
        If True, convergence information is printed.  For the default
        'lbfgs' ``solver``, disp controls the frequency of the output during
        the iterations. disp < 0 means no output in this case.

    callback : callable, optional (default=None)
        Called after each iteration as callback(xk) where xk is the current
        parameter vector.

    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If ``suppress_warnings``
        is True, all of these warnings will be squelched.


    Notes
    -----
    * Since the ``ARIMA`` class currently wraps ``statsmodels.tsa.arima_model.ARIMA``, which does not
      provide support for seasonality, the only way to fit seasonal ARIMAs is to manually lag/pre-process
      your data appropriately. This might change in the future. [2]

    * After the model fit, many more methods will become available to the fitted model (i.e., :func:`pvalues`,
      :func:`params`, etc.). These are delegate methods which wrap the internal ARIMA results instance.


    References
    ----------
    [1] https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    [2] http://www.statsmodels.org/0.6.1/generated/statsmodels.tsa.arima_model.ARIMA.html
    """
    def __init__(self, order, start_params=None, trend='c', method="css-mle", transparams=True,
                 solver='lbfgs', maxiter=50, disp=0, callback=None, suppress_warnings=False):
        super(ARIMA, self).__init__()

        self.order = order
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.transparams = transparams
        self.solver = solver
        self.maxiter = maxiter
        self.disp = disp
        self.callback = callback
        self.suppress_warnings = suppress_warnings

    def fit(self, y, exogenous=None, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations.

        Parameters
        ----------
        y : array-like, shape=(n_samples,)
            The time-series on which to fit the ARIMA.

        exogenous : array-like, shape=[n_samples, n_features], optional (default=None)
            An optional array of exogenous variables. This should not
            include a constant or trend.
        """
        y = check_array(y, ensure_2d=False, force_all_finite=False, copy=True, dtype=DTYPE)

        # if exog was included, check the array...
        if exogenous is not None:
            exogenous = check_array(exogenous, ensure_2d=True, force_all_finite=False,
                                    copy=False, dtype=DTYPE)

        # create and fit the statsmodels ARIMA
        self.arima_ = _ARIMA(endog=y, order=self.order, missing='none', exog=exogenous, dates=None, freq=None)

        def _fit_wrapper():
            return self.arima_.fit(start_params=self.start_params,
                                   trend=self.trend, method=self.method,
                                   transparams=self.transparams,
                                   solver=self.solver, maxiter=self.maxiter,
                                   disp=self.disp, callback=self.callback, **fit_args)

        # sometimes too many warnings...
        if self.suppress_warnings:
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                self.arima_res_ = _fit_wrapper()
        else:
            self.arima_res_ = _fit_wrapper()

        return self

    def predict(self, n_periods=10, exogenous=None, alpha=0.05,
                include_std_err=False, include_conf_int=False):
        """Generate predictions (forecasts) ``n_periods`` in the future. Note that unless
        ``include_std_err`` or ``include_conf_int`` are True, only the forecast
        array will be returned (otherwise, a tuple with the corresponding elements
        will be returned).


        Parameters
        ----------
        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        exogenous : array-like, shape=[n_samples, n_features], optional (default=None)
            An optional array of exogenous variables. This should not
            include a constant or trend.

        alpha : float
            The confidence intervals for the forecasts are (1 - alpha) %

        include_std_err : bool, optional (default=False)
            Whether to return the array of standard errors.

        include_conf_int : bool, optional (default=False)
            Whether to return the array of confidence intervals.


        Returns
        -------
        forecasts : array-like, shape=(n_periods,)
            The array of fore-casted values.

        stderr : array-like, shape=(n_periods,)
            The array of standard errors (NOTE this is only returned
            if ``include_std_err`` is True).

        conf_int : array-like, shape=(n_periods, 2)
            The array of confidence intervals (NOTE this is only returned
            if ``include_conf_int`` is True).
        """
        check_is_fitted(self, 'arima_res_')

        # use the results wrapper to predict so it injects its own params
        # (also if I was 0, ARMA will not have a forecast method natively)
        f, s, c = self.arima_res_.forecast(steps=n_periods, exog=exogenous, alpha=alpha)

        # different variants of the stats someone might want returned...
        if include_std_err and include_conf_int:
            return f, s, c
        elif include_conf_int:
            return f, c
        elif include_std_err:
            return f, s
        return f

    def fit_predict(self, y, exogenous=None, n_periods=10, alpha=0.05,
                    include_std_err=False, include_conf_int=False, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations, and then
        generate predictions.

        Parameters
        ----------
        y : array-like, shape=(n_samples,)
            The time-series on which to fit the ARIMA.

        exogenous : array-like, optional (default=None)
            An optional array of exogenous variables. This should not
            include a constant or trend.

        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        alpha : float
            The confidence intervals for the forecasts are (1 - alpha) %

        include_std_err : bool, optional (default=False)
            Whether to return the array of standard errors.

        include_conf_int : bool, optional (default=False)
            Whether to return the array of confidence intervals.

        fit_args : dict, optional (default=None)
            Any keyword args to pass to the fit method.
        """
        self.fit(y, exogenous, **fit_args)
        return self.predict(n_periods=n_periods, exogenous=exogenous, alpha=alpha,
                            include_std_err=include_std_err, include_conf_int=include_conf_int)

    @if_delegate_has_method('arima_res_')
    def aic(self):
        """Get the AIC, the Akaine Information Criterion:

            -2 * llf + 2 * df_model

        Where ``df_model`` (the number of degrees of freedom in the model)
        includes all AR parameters, MA parameters, constant terms parameters
        on constant terms and the variance.

        Returns
        -------
        aic : float
            The AIC
        """
        return self.arima_res_.aic

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

            (1 - arparams[0] * z - arparams[1] * z^2 - ... - arparams[p-1] * z^k_ar) = 0

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

            -2 * llf + log(nobs) * df_model

        Where if the model is fit using conditional sum of squares, the
        number of observations ``nobs`` does not include the ``p`` pre-sample
        observations.

        Returns
        -------
        bse : float
            The BIC
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
    def df_model(self):
        """Get the model degrees of freedom:

            k_exog + k_trend + k_ar + k_ma

        Returns
        -------
        df_model : array-like
            The degrees of freedom.
        """
        return self.arima_res_.df_model

    @if_delegate_has_method('arima_res_')
    def df_resid(self):
        """Get the residual degrees of freedom:

            nobs - df_model

        Returns
        -------
        df_resid : array-like
            The residual degrees of freedom.
        """
        return self.arima_res_.df_resid

    @if_delegate_has_method('arima_res_')
    def hqic(self):
        """Get the Hannan-Quinn Information Criterion:

            -2 * llf + 2 * (`df_model`) * log(log(nobs))

        Like :func:`bic` if the model is fit using conditional sum of squares then
        the ``k_ar`` pre-sample observations are not counted in ``nobs``.

        Returns
        -------
        hqic : float
            The HQIC
        """
        return self.arima_res_.hqic

    @if_delegate_has_method('arima_res_')
    def k_ar(self):
        """Get the number of AR coefficients in the model.

        Returns
        -------
        k_ar : int
            The number of AR coefficients.
        """
        return self.arima_res_.k_ar

    @if_delegate_has_method('arima_res_')
    def k_exog(self):
        """Get the number of exogenous variables included in the model. Does not
        include the constant.

        Returns
        -------
        k_exog : int
            The number of features in the exogenous variables.
        """
        return self.arima_res_.k_exog

    @if_delegate_has_method('arima_res_')
    def k_ma(self):
        """Get the number of MA coefficients in the model.

        Returns
        -------
        k_ma : int
            The number of MA coefficients.
        """
        return self.arima_res_.k_ma

    @if_delegate_has_method('arima_')
    def loglike(self):
        """Get the log-likelihood

        Returns
        -------
        loglike : float
            The log likelihood
        """
        return self.arima_.loglike

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

            (1 + maparams[0] * z + maparams[1] * z^2 + ... + maparams[q-1] * z^q) = 0

        Stability requires that the roots in modules lie outside the unit
        circle.

        Returns
        -------
        maroots : array-like
            The MA roots.
        """
        return self.arima_res_.maroots

    @if_delegate_has_method('arima_res_')
    def params(self):
        """Get the parameters of the model. The order of variables is the trend
        coefficients and the :func:`k_exog` exogenous coefficients, then the
        :func:`k_ar` AR coefficients, and finally the :func:`k_ma` MA coefficients.

        Returns
        -------
        params : array-like
            The parameters of the model.
        """
        return self.arima_res_.params

    @if_delegate_has_method('arima_res_')
    def pvalues(self):
        """Get the p-values associated with the t-values of the coefficients. Note
        that the coefficients are assumed to have a Student's T distribution.

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
        using 'css' then the residuals are obtained via ``scipy.signal.lfilter``
        adjusted such that the first :func:`k_ma` residuals are zero. These zero
        residuals are not returned.

        Returns
        -------
        resid : array-like
            The model residuals.
        """
        return self.arima_res_.resid

    @if_delegate_has_method('arima_res_')
    def sigma2(self):
        """Get the variance of the residuals. If the model is fit by 'css',
        sigma2 = ssr/nobs, where ssr is the sum of squared residuals. If
        the model is fit by 'mle', then sigma2 = 1/nobs * sum(v^2 / F)
        where v is the one-step forecast error and F is the forecast error
        variance.

        Returns
        -------
        sigma2 : float
            The variance of the residuals
        """
        return self.arima_res_.sigma2
