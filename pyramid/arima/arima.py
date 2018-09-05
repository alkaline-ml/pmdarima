# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# A much more user-friendly wrapper to the statsmodels ARIMA.
# Mimics the familiar sklearn interface.

from __future__ import print_function, absolute_import, division

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_array, check_is_fitted, \
    column_or_1d as c1d

from statsmodels.tsa.arima_model import ARIMA as _ARIMA
from statsmodels.tsa.base.tsa_model import TimeSeriesModelResults
from statsmodels import api as sm

import numpy as np
import datetime
import warnings
import os

from ..compat.numpy import DTYPE  # DTYPE for arrays
from ..compat.python import long, safe_mkdirs
from ..compat import statsmodels as sm_compat
from ..utils import get_callable, if_has_delegate
from ..utils.array import diff
from .._config import PYRAMID_ARIMA_CACHE, PICKLE_HASH_PATTERN

# Get the version
import pyramid

__all__ = [
    'ARIMA'
]

VALID_SCORING = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error
}


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


class ARIMA(BaseEstimator):
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
        The number of examples from the tail of the time series to hold out
        and use as validation examples. The model will not be fit on these
        samples, but the observations will be added into the model's ``endog``
        and ``exog`` arrays so that future forecast values originate from the
        end of the endogenous vector. See :func:`add_new_observations`.

        For instance::

            y = [0, 1, 2, 3, 4, 5, 6]
            out_of_sample_size = 2

            > Fit on: [0, 1, 2, 3, 4]
            > Score on: [5, 6]
            > Append [5, 6] to end of self.arima_res_.data.endog values

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

        # XXX: This isn't actually required--sklearn doesn't need a super call
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

        **fit_args : dict or kwargs
            Any keyword arguments to pass to the statsmodels ARIMA fit.
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

            # This also means we have to address the exogenous matrix
            if exogenous is not None:
                cv_exog = exogenous[-cv:, :]
                exogenous = exogenous[:-cv, :]

        # This wrapper is used for fitting either an ARIMA or a SARIMAX
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
        sm_compat.bind_df_model(fit, self.arima_res_)

        # if the model is fit with an exogenous array, it must
        # be predicted with one as well.
        self.fit_with_exog_ = exogenous is not None

        # now make a forecast if we're validating to compute the
        # out-of-sample score
        if cv_samples is not None:
            # get the predictions (use self.predict, which calls forecast
            # from statsmodels internally)
            pred = self.predict(n_periods=cv, exogenous=cv_exog)
            self.oob_ = scoring(cv_samples, pred, **self.scoring_args)

            # If we compute out of sample scores, we have to now update the
            # observed time points so future forecasts originate from the end
            # of our y vec
            self.add_new_observations(cv_samples, cv_exog)
        else:
            self.oob_ = np.nan

        # Save nobs since we might change it later if using OOB
        self.nobs_ = y.shape[0]

        # As of version 0.7.2, start saving the version with the model so
        # we can track changes over time.
        self.pkg_version_ = pyramid.__version__
        return self

    def _check_exog(self, exogenous):
        # if we fit with exog, make sure one was passed, or else fail out:
        if self.fit_with_exog_:
            if exogenous is None:
                raise ValueError('When an ARIMA is fit with an exogenous '
                                 'array, it must also be provided one for '
                                 'predicting or updating observations.')
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

    def predict(self, n_periods=10, exogenous=None,
                return_conf_int=False, alpha=0.05):
        """Generate predictions (forecasts) ``n_periods`` in the future.
        Note that if ``exogenous`` variables were used in the model fit, they
        will be expected for the predict procedure and will fail otherwise.

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
        if not isinstance(n_periods, (int, long)):
            raise TypeError("n_periods must be an int or a long")

        # if we fit with exog, make sure one was passed:
        exogenous = self._check_exog(exogenous)  # type: np.ndarray
        if exogenous is not None and exogenous.shape[0] != n_periods:
            raise ValueError('Exogenous array dims (n_rows) != n_periods')

        # ARIMA predicts differently...
        if self.seasonal_order is None:
            # use the results wrapper to predict so it injects its own params
            # (also if I was 0, ARMA will not have a forecast method natively)
            f, _, conf_int = self.arima_res_.forecast(
                steps=n_periods, exog=exogenous, alpha=alpha)
        else:  # SARIMAX
            # Unfortunately, SARIMAX does not really provide a nice way to get
            # the confidence intervals out of the box, so we have to perform
            # the get_prediction code here and unpack the confidence intervals
            # manually.
            # f = self.arima_res_.forecast(steps=n_periods, exog=exogenous)
            arima = self.arima_res_
            end = arima.nobs + n_periods - 1
            results = arima.get_prediction(start=arima.nobs, end=end,
                                           exog=exogenous)
            f = results.predicted_mean
            conf_int = results.conf_int(alpha=alpha)

        if return_conf_int:
            # The confidence intervals may be a Pandas frame if it comes from
            # SARIMAX & we want Numpy. We will to duck type it so we don't add
            # new explicit requirements for the package
            return f, check_array(conf_int)  # duck type for pd.DataFrame
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

        fit_args : dict or kwargs, optional (default=None)
            Any keyword args to pass to the fit method.
        """
        self.fit(y, exogenous, **fit_args)
        return self.predict(n_periods=n_periods, exogenous=exogenous)

    def _get_pickle_hash_file(self):
        # Mmmm, pickle hash...
        return PICKLE_HASH_PATTERN % (
            # cannot use ':' in Windows file names. Whoops!
            str(datetime.datetime.now()).replace(' ', '_').replace(':', '-'),
            ''.join([str(e) for e in self.order]),
            hash(self))

    def __getstate__(self):
        """I am being pickled..."""

        # In versions <0.9.0, if this already contains a pointer to a
        # "saved state" model, we deleted that model and replaced it with the
        # new one.
        # In version >= v0.9.0, we keep the old model around, since that's how
        # the user expects it should probably work (otherwise unpickling the
        # previous state of the model would raise an OSError).
        # loc = self.__dict__.get('tmp_pkl_', None)
        # if loc is not None:
        #     os.unlink(loc)

        # get the new location for where to save the results
        new_loc = self._get_pickle_hash_file()
        cwd = os.path.abspath(os.getcwd())

        # check that the cache folder exists, and if not, make it.
        cache_loc = os.path.join(cwd, PYRAMID_ARIMA_CACHE)
        safe_mkdirs(cache_loc)

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

        # Warn for unpickling a different version's model
        self._warn_for_older_version()
        return self

    def _warn_for_older_version(self):
        # Added in v0.8.1 - check for the version pickled under and warn
        # if it's different from the current version
        do_warn = False
        modl_version = None
        this_version = pyramid.__version__

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
                          "Pyramid (%s). This could cause unforeseen behavior."
                          % (modl_version, this_version), UserWarning)

    def _clear_cached_state(self):
        # when fit in an auto-arima, a lot of cached .pmdpkl files
        # are generated if fit in parallel... this removes the tmp file
        loc = self.__dict__.get('tmp_pkl_', None)
        if loc is not None:
            os.unlink(loc)

    def add_new_observations(self, y, exogenous=None):
        """Update the endog/exog samples after a model fit.

        After fitting your model and creating forecasts, you're going
        to need to attach new samples to the data you fit on. These are
        used to compute new forecasts (but using the same estimated
        parameters).

        Parameters
        ----------
        y : array-like or iterable, shape=(n_samples,)
            The time-series data to add to the endogenous samples on which the
            ``ARIMA`` estimator was previously fit. This may either be a Pandas
            ``Series`` object or a numpy array. This should be a one-
            dimensional array of finite floats.

        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d array of exogenous variables. If the model was
            fit with an exogenous array of covariates, it will be required for
            updating the observed values.

        Notes
        -----
        This does not constitute re-fitting, as the model params do not
        change, so do not use this in place of periodically refreshing the
        model. Use it only to add new observed values from which to forecast
        new values.
        """
        check_is_fitted(self, 'arima_res_')
        model_res = self.arima_res_

        # validate the new samples to add
        y = c1d(check_array(y, ensure_2d=False, force_all_finite=False,
                            copy=True, dtype=DTYPE))  # type: np.ndarray
        n_samples = y.shape[0]

        # if exogenous is None and new exog provided, or vice versa, raise
        exogenous = self._check_exog(exogenous)  # type: np.ndarray

        # ensure the k_exog matches
        if exogenous is not None:
            k_exog = model_res.model.k_exog
            n_exog, exog_dim = exogenous.shape

            if exogenous.shape[1] != k_exog:
                raise ValueError("Dim mismatch in fit exogenous (%i) and new "
                                 "exogenous (%i)" % (k_exog, exog_dim))

            # make sure the number of samples in exogenous match the number
            # of samples in the endog
            if n_exog != n_samples:
                raise ValueError("Dim mismatch in n_samples "
                                 "(endog=%i, exog=%i)"
                                 % (n_samples, n_exog))

        # difference the y array to concatenate (now n_samples - d)
        d = self.order[1]

        # first concatenate the original data (might be 2d or 1d)
        y = _append_to_endog(model_res.data.endog, y)

        # Now create the new exogenous.
        if exogenous is not None:
            # Concatenate
            exog = np.concatenate((model_res.data.exog, exogenous))
        else:
            # Just so it's in the namespace
            exog = None

        # Update the arrays in the data class. The statsmodels ARIMA class
        # stores the values a bit differently than it does in the SARIMAX
        # class...
        sarimax = self.seasonal_order is not None
        if not sarimax:  # ARIMA or ARMA

            # Set the endog in two places. The undifferenced array in the
            # model_res.data, and the differenced array in the model_res.model
            model_res.data.endog = c1d(y)  # type: np.ndarray

            # The model endog is stored differently in the ARIMA class than
            # in the SARIMAX class, where the ARIMA actually stores the diffed
            # array. However, ARMA does not (and we cannot diff for d < 1).
            do_diff = d > 0
            if do_diff:  # ARIMA
                y_diffed = diff(y, d)
            else:  # ARMA
                y_diffed = y

            # This changes the length of the array!
            model_res.model.endog = y_diffed

            # Set the model result nobs (must be the differenced shape!)
            model_res.nobs = y_diffed.shape[0]

            # Set the exogenous
            if exog is not None:
                # Set in data class (this is NOT differenced, unlike the
                # model data)
                model_res.data.exog = exog

                # Difference and add intercept, then add to model class
                k_intercept = (model_res.model.exog.shape[1] -
                               exogenous.shape[1])
                exog_diff = exog[d:, :]
                intercept = np.ones((exog_diff.shape[0], k_intercept))
                exog_diff = np.hstack((intercept, exog_diff))

                # set in the model itself
                model_res.model.exog = exog_diff

            else:
                # Otherwise we STILL have to set the exogenous array as an
                # intercept in the model class for both ARMA and ARIMA.
                # Make sure to use y_diffed in case d > 0, since the exog
                # array will be multiplied by the endog at some point and we
                # need the dimensions to match (Issue #30)
                model_res.model.exog = np.ones((y_diffed.shape[0], 1))

        else:  # SARIMAX
            # The model endog is stored differently in the ARIMA class than
            # in the SARIMAX class, where the SARIMAX is a 2d (n x 1) array
            # that is NOT diffed. We also handle this piece a bit differently..
            # In the SARIMAX class, statsmodels creates a "pseudo new" model
            # with the same parameters for forecasting, and we'll do the same.
            model_kwargs = model_res._init_kwds.copy()

            if exog is not None:
                model_kwargs['exog'] = exog

            # Create the pseudo "new" model and set its parameters with the
            # existing model fit parameters
            new_model = sm.tsa.statespace.SARIMAX(endog=y, **model_kwargs)
            new_model.update(model_res.params)

            # Point the arima result to the new model
            self.arima_res_.model = new_model

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
        # FIXME:
        # this code should really be added to statsmodels. Rewrite
        # this function to reflect other metric implementations if/when
        # statsmodels incorporates AICc

        aic = self.arima_res_.aic
        nobs = self.nobs_
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
