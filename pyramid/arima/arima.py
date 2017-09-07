# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# A much more user-friendly wrapper to the statsmodels ARIMA.
# Mimics the familiar sklearn interface.

from __future__ import print_function, absolute_import, division

from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      column_or_1d as c1d)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.metaestimators import if_delegate_has_method
from statsmodels.tsa.arima_model import ARIMA as _ARIMA
from statsmodels.tsa.base.tsa_model import TimeSeriesModelResults
from statsmodels import api as sm
import numpy as np
import datetime
import warnings
import os

# DTYPE for arrays
from ..compat.numpy import DTYPE
from ..utils import get_callable, if_has_delegate

__all__ = [
    'ARIMA'
]

VALID_SCORING = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error
}


class ARIMA(BaseEstimator):
    """An ARIMA, or autoregressive integrated moving average, is a
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

    seasonal_order : array-like, shape=(4,), optional (default=None)
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

    transparams : bool, optional (default=True)
        Whehter or not to transform the parameters to ensure stationarity.
        Uses the transformation suggested in Jones (1980).  If False,
        no checking for stationarity or invertibility is done.

    method : str, one of {'css-mle','mle','css'}, optional (default=None)
        This is the loglikelihood to maximize.  If "css-mle", the
        conditional sum of squares likelihood is maximized and its values
        are used as starting values for the computation of the exact
        likelihood via the Kalman filter.  If "mle", the exact likelihood
        is maximized via the Kalman Filter.  If "css" the conditional sum
        of squares likelihood is maximized.  All three methods use
        `start_params` as starting parameters.  See above for more
        information. If fitting a seasonal ARIMA, the default is 'lbfgs'

    trend : str or iterable, optional (default='c')
        Parameter controlling the deterministic trend polynomial :math:`A(t)`.
        Can be specified as a string where 'c' indicates a constant (i.e. a
        degree zero component of the trend polynomial), 't' indicates a
        linear trend with time, and 'ct' is both. Can also be specified as an
        iterable defining the polynomial as in ``numpy.poly1d``, where
        ``[1,1,0,1]`` would denote :math:`a + bt + ct^3`.

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
        parameter vector. This is only used in non-seasonal ARIMA models.

    suppress_warnings : bool, optional (default=False)
        Many warnings might be thrown inside of statsmodels. If
        ``suppress_warnings`` is True, all of these warnings will be squelched.

    out_of_sample_size : int, optional (default=0)
        The number of examples from the tail of the time series to use as
        validation examples.

    scoring : str, optional (default='mse')
        If performing validation (i.e., if ``out_of_sample_size`` > 0), the
        metric to use for scoring the out-of-sample data. One of {'mse', 'mae'}

    scoring_args : dict, optional (default=None)
        A dictionary of key-word arguments to be passed to the
        ``scoring`` metric.

    Notes
    -----
    * Since the ``ARIMA`` class currently wraps
      ``statsmodels.tsa.arima_model.ARIMA``, which does not provide support
      for seasonality, the only way to fit seasonal ARIMAs is to manually
      lag/pre-process your data appropriately. This might change in
      the future. [2]

    * After the model fit, many more methods will become available to the
      fitted model (i.e., :func:`pvalues`, :func:`params`, etc.). These are
      delegate methods which wrap the internal ARIMA results instance.

    See Also
    --------
    :func:`pyramid.arima.auto_arima`

    References
    ----------
    .. [1] https://wikipedia.org/wiki/Autoregressive_integrated_moving_average

    .. [2] Statsmodels ARIMA documentation: http://bit.ly/2wc9Ra8
    """
    def __init__(self, order, seasonal_order=None, start_params=None, trend='c',
                 method=None, transparams=True, solver='lbfgs', maxiter=50,
                 disp=0, callback=None, suppress_warnings=False,
                 out_of_sample_size=0, scoring='mse', scoring_args=None):
        super(ARIMA, self).__init__()

        self.order = order
        self.seasonal_order = seasonal_order
        self.start_params = start_params
        self.trend = trend
        self.method = method
        self.transparams = transparams
        self.solver = solver
        self.maxiter = maxiter
        self.disp = disp
        self.callback = callback
        self.suppress_warnings = suppress_warnings
        self.out_of_sample_size = out_of_sample_size
        self.scoring = scoring
        self.scoring_args = dict() if not scoring_args else scoring_args

    def fit(self, y, exogenous=None, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations with an
        optional matrix of ``exogenous`` variables.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series to which to fit the ``ARIMA`` estimator. This may
            either be a Pandas ``Series`` object (statsmodels can internally
            use the dates in the index), or a numpy array. This should be a
            one-dimensional array of floats, and should not contain any
            ``np.nan`` or ``np.inf`` values.

        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.
        """
        y = c1d(check_array(y, ensure_2d=False, force_all_finite=False,
                            copy=True, dtype=DTYPE))  # type: np.ndarray
        n_samples = y.shape[0]

        # if exog was included, check the array...
        if exogenous is not None:
            exogenous = check_array(exogenous, ensure_2d=True,
                                    force_all_finite=False,
                                    copy=False, dtype=DTYPE)

        # determine the CV args, if any
        cv = self.out_of_sample_size
        scoring = get_callable(self.scoring, VALID_SCORING)

        # don't allow negative, don't allow > n_samples
        cv = max(min(cv, n_samples), 0)

        def _fit_wrapper():
            # these might change depending on which one
            method = self.method

            # if not seasonal:
            if self.seasonal_order is None:
                if method is None:
                    method = "css-mle"

                # create the statsmodels ARIMA
                arima = _ARIMA(endog=y, order=self.order, missing='none',
                               exog=exogenous, dates=None, freq=None)

                # there's currently a bug in the ARIMA model where on pickling
                # it tries to acquire an attribute called
                # 'self.{dates|freq|missing}', but they do not exist as class
                # attrs! They're passed up to TimeSeriesModel in base, but
                # are never set. So we inject them here so as not to get an
                # AttributeError later. (see http://bit.ly/2f7SkKH)
                for attr, val in (('dates', None), ('freq', None),
                                  ('missing', 'none')):
                    if not hasattr(arima, attr):
                        setattr(arima, attr, val)
            else:
                if method is None:
                    method = 'lbfgs'

                # create the SARIMAX
                arima = sm.tsa.statespace.SARIMAX(
                    endog=y, exog=exogenous, order=self.order,
                    seasonal_order=self.seasonal_order, trend=self.trend,
                    enforce_stationarity=self.transparams)

            # actually fit the model, now...
            return arima, arima.fit(start_params=self.start_params,
                                    trend=self.trend, method=method,
                                    transparams=self.transparams,
                                    solver=self.solver, maxiter=self.maxiter,
                                    disp=self.disp, callback=self.callback,
                                    **fit_args)

        # sometimes too many warnings...
        if self.suppress_warnings:
            with warnings.catch_warnings(record=False):
                warnings.simplefilter('ignore')
                fit, self.arima_res_ = _fit_wrapper()
        else:
            fit, self.arima_res_ = _fit_wrapper()

        # Set df_model attribute for SARIMAXResults object
        if not hasattr(self.arima_res_, 'df_model'):
            df_model = fit.k_exog + fit.k_trend + fit.k_ar + \
                       fit.k_ma + fit.k_seasonal_ar + fit.k_seasonal_ma
            setattr(self.arima_res_, 'df_model', df_model)

        # if the model is fit with an exogenous array, it must
        # be predicted with one as well.
        self.fit_with_exog_ = exogenous is not None

        # now make a prediction if we're validating
        # to save the out-of-sample value
        if cv > 0:
            # get the predictions
            pred = self.arima_res_.predict(exog=exogenous, typ='linear')[-cv:]
            self.oob_ = scoring(y[-cv:], pred, **self.scoring_args)
        else:
            self.oob_ = np.nan

        return self

    def _check_exog(self, exogenous):
        # if we fit with exog, make sure one was passed, or else fail out:
        if self.fit_with_exog_:
            if exogenous is None:
                raise ValueError('When an ARIMA is fit with an exogenous '
                                 'array, it must be provided one for '
                                 'predicting (either in- OR out-of-sample).')
            else:
                return check_array(exogenous, ensure_2d=True,
                                   force_all_finite=True, dtype=DTYPE)
        return None

    def predict_in_sample(self, exogenous=None, start=None,
                          end=None, dynamic=False):
        """Generate in-sample predictions from the fit ARIMA model. This can
        be useful when wanting to visualize the fit, and qualitatively inspect
        the efficacy of the model, or when wanting to compute the residuals
        of the model.

        Parameters
        ----------
        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        start : int, optional (default=None)
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start.

        end : int, optional (default=None)
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start.

        dynamic : bool, optional
            The `dynamic` keyword affects in-sample prediction. If dynamic
            is False, then the in-sample lagged values are used for
            prediction. If `dynamic` is True, then in-sample forecasts are
            used in place of lagged dependent variables. The first forecasted
            value is `start`.

        Returns
        -------
        predict : array
            The predicted values.
        """
        check_is_fitted(self, 'arima_res_')

        # if we fit with exog, make sure one was passed:
        exogenous = self._check_exog(exogenous)  # type: np.ndarray
        return self.arima_res_.predict(exog=exogenous, start=start,
                                       end=end, dynamic=dynamic)

    def predict(self, n_periods=10, exogenous=None):
        """Generate predictions (forecasts) ``n_periods`` in the future.
        Note that unless ``include_std_err`` or ``include_conf_int`` are True,
        only the forecast array will be returned (otherwise, a tuple with the
        corresponding elements will be returned).

        Parameters
        ----------
        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        Returns
        -------
        forecasts : array-like, shape=(n_periods,)
            The array of fore-casted values.
        """
        check_is_fitted(self, 'arima_res_')

        # if we fit with exog, make sure one was passed:
        exogenous = self._check_exog(exogenous)  # type: np.ndarray
        if exogenous is not None and exogenous.shape[0] != n_periods:
            raise ValueError('Exogenous array dims (n_rows) != n_periods')

        # ARIMA predicts differently...
        if self.seasonal_order is None:
            # use the results wrapper to predict so it injects its own params
            # (also if I was 0, ARMA will not have a forecast method natively)
            f, _, _ = self.arima_res_.forecast(steps=n_periods, exog=exogenous)
        else:
            f = self.arima_res_.forecast(steps=n_periods, exog=exogenous)

        # different variants of the stats someone might want returned...
        return f

    def fit_predict(self, y, exogenous=None, n_periods=10, **fit_args):
        """Fit an ARIMA to a vector, ``y``, of observations with an
        optional matrix of ``exogenous`` variables, and then generate
        predictions.

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series to which to fit the ``ARIMA`` estimator. This may
            either be a Pandas ``Series`` object (statsmodels can internally
            use the dates in the index), or a numpy array. This should be a
            one-dimensional array of floats, and should not contain any
            ``np.nan`` or ``np.inf`` values.

        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must be provided
            exogenous features for making predictions.

        n_periods : int, optional (default=10)
            The number of periods in the future to forecast.

        fit_args : dict, optional (default=None)
            Any keyword args to pass to the fit method.
        """
        self.fit(y, exogenous, **fit_args)
        return self.predict(n_periods=n_periods, exogenous=exogenous)

    @staticmethod
    def _get_cache_folder():
        return '.pyramid-cache'

    def _get_pickle_hash_file(self):
        # Mmmm, pickle hash...
        return '%s-%s-%i.pmdpkl' % (
            # cannot use ':' in Windows file names. Whoops!
            str(datetime.datetime.now()).replace(' ', '_').replace(':', '-'),
            ''.join([str(e) for e in self.order]),
            hash(self))

    def __getstate__(self):
        """I am being pickled..."""
        loc = self.__dict__.get('tmp_pkl_', None)

        # if this already contains a pointer to a "saved state" model,
        # delete that model and replace it with a new one
        if loc is not None:
            os.unlink(loc)

        # get the new location for where to save the results
        new_loc = self._get_pickle_hash_file()
        cwd = os.path.abspath(os.getcwd())

        # check that the cache folder exists, and if not, make it.
        cache_loc = os.path.join(cwd, self._get_cache_folder())
        try:
            os.makedirs(cache_loc)
        # since this is a race condition, just try to make it
        except OSError as e:
            if e.errno != 17:
                raise

        # now create the full path with the cache folder
        new_loc = os.path.join(cache_loc, new_loc)

        # save the results - but only if it's fit...
        if hasattr(self, 'arima_res_'):
            # statsmodels result views work by caching metrics. If they
            # are not cached prior to pickling, we might hit issues. This is
            # a bug documented here:
            # https://github.com/statsmodels/statsmodels/issues/3290
            self.arima_res_.summary()
            self.arima_res_.save(fname=new_loc)  # , remove_data=False)

            # point to the location of the saved MLE model
            self.tmp_pkl_ = new_loc

        return self.__dict__

    def __setstate__(self, state):
        # I am being unpickled...
        self.__dict__ = state

        # re-set the results class
        loc = state.get('tmp_pkl_', None)
        if loc is not None:
            try:
                self.arima_res_ = TimeSeriesModelResults.load(loc)
            except:
                raise OSError('Could not read saved model state from %s. '
                              'Does it still exist?' % loc)

        return self

    def _clear_cached_state(self):
        # when fit in an auto-arima, a lot of cached .pmdpkl files
        # are generated if fit in parallel... this removes the tmp file
        loc = self.__dict__.get('tmp_pkl_', None)
        if loc is not None:
            os.unlink(loc)

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
        # TODO: this code should really be added to statsmodels. Rewrite
        #       this function to reflect other metric implementations if/when
        #       statsmodels incorporates AICc
        aic = self.arima_res_.aic
        nobs = self.arima_res_.nobs
        df_model = self.arima_res_.df_model + 1  # add one for constant term
        return aic + 2. * df_model * (nobs / (nobs - df_model - 1.) - 1.)

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
    def df_model(self):
        """The model degrees of freedom: ``k_exog`` + ``k_trend`` +
        ``k_ar`` + ``k_ma``.

        Returns
        -------
        df_model : array-like
            The degrees of freedom in the model.
        """

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
