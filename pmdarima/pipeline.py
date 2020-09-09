# -*- coding: utf-8 -*-

from itertools import islice
import pandas as pd
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.utils.metaestimators import if_delegate_has_method

from .base import BaseARIMA
from .preprocessing.base import BaseTransformer
from .preprocessing.endog.base import BaseEndogTransformer
from .preprocessing.exog.base import BaseExogTransformer, BaseExogFeaturizer
from .utils import check_endog
from .compat import DTYPE, check_is_fitted
from .compat import pmdarima as pm_compat

__all__ = ['Pipeline']


def _warn_for_deprecated(**kwargs):
    # TODO: remove this in the future
    for k in ('typ',):
        if kwargs.pop(k, None):
            warnings.warn("'%s' is deprecated and will be removed in a future "
                          "release" % k,
                          DeprecationWarning)
    return kwargs


class Pipeline(BaseEstimator):
    """A pipeline of transformers with an optional final estimator stage

    The pipeline object chains together an arbitrary number of named, ordered
    transformations, passing the output from one as the input to the next. As
    the last stage, an ``ARIMA`` or ``AutoARIMA`` object will be fit. This
    pipeline takes after the scikit-learn ``sklearn.Pipeline`` object, which
    behaves similarly but does not share the same time-series interface
    that ``pmdarima`` follows.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a `'__'`, as in the example
    below.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an ARIMA or AutoARIMA estimator.

    Examples
    --------
    >>> from pmdarima.datasets import load_wineind
    >>> from pmdarima.arima import AutoARIMA
    >>> from pmdarima.pipeline import Pipeline
    >>> from pmdarima.preprocessing import FourierFeaturizer
    >>>
    >>> wineind = load_wineind()
    >>> pipeline = Pipeline([
    ...     ("fourier", FourierFeaturizer(m=12, k=3)),
    ...     ("arima", AutoARIMA(seasonal=False, stepwise=True,
    ...                         suppress_warnings=True,
    ...                         error_action='ignore'))
    ... ])
    >>> pipeline.fit(wineind)
    Pipeline(steps=[('fourier', FourierFeaturizer(k=3, m=12)),
                    ('arima', AutoARIMA(D=None, alpha=0.05, callback=None,
                                        d=None, disp=0, error_action='ignore',
                                        information_criterion='aic', m=1,
                                        max_D=1, max_P=2, max_Q=2, max_d=2,
                                        max_order=10, max_p=5, max_q=5,
                                        maxiter=None, method=None,
                                        n_fits=10, n...s_warnings=True,
                                        test='kpss', trace=False,
                                        transparams=True, trend=None,
                                        with_intercept=True))])
    """

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_names(self, names):
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))
        invalid_names = set(names).intersection(self.get_params(deep=False))
        if invalid_names:
            raise ValueError('Estimator names conflict with constructor '
                             'arguments: {0!r}'.format(sorted(invalid_names)))
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            # Transformers must be endog/exog transformers
            if not isinstance(t, BaseTransformer):
                raise TypeError("All intermediate steps should be "
                                "instances of BaseTransformer, but "
                                "'%s' (type %s) is not" % (t, type(t)))

        if not isinstance(estimator, BaseARIMA):
            raise TypeError(
                "Last step of Pipeline should be of type BaseARIMA. "
                "'%s' (type %s) isn't" % (estimator, type(estimator)))

        # Shallow copy
        return list(self.steps)

    def _iter(self, with_final=True):
        """
        Generate (name, trans) tuples
        """
        # MUST have called fit first
        stop = len(self.steps_)
        if not with_final:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps_, 0, stop)):
            yield idx, name, trans

    def _get_kwargs(self, **params):
        params_steps = {name: {} for name, step in self.steps
                        if step is not None}
        for pname, pval in params.items():
            step, param = pname.split('__', 1)
            params_steps[step][param] = pval
        return params_steps

    def __len__(self):
        """
        Returns the length of the Pipeline
        """
        return len(self.steps)

    @property
    def named_steps(self):
        """Map the steps to a dictionary"""
        return dict(self.steps)

    @property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return estimator

    def _check_n_periods(self, n_periods, exog):
        # if exog was passed in, we have to use that as n_periods
        if exog is not None and len(exog) != n_periods:
            n_periods = len(exog)
        return n_periods

    def fit(self, y, X=None, **fit_kwargs):
        """Fit the pipeline of transformers and the ARIMA model

        Chain the time-series and X array through a series of
        transformations, fitting each stage along the way, finally fitting an
        ARIMA or AutoARIMA model.

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

        **fit_kwargs : keyword args
            Extra keyword arguments used for each stage's ``fit`` stage.
            Similar to scikit-learn pipeline keyword args, the keys are
            compound, comprised of the stage name and the argument name
            separated by a "__". For instance, if fitting an ARIMA in stage
            "arima", your kwargs may resemble::

                {"arima__maxiter": 10}
        """
        # Shallow copy
        steps = self.steps_ = self._validate_steps()

        # Temporary shim until we remove `exogenous` support completely
        X, fit_kwargs = pm_compat.get_X(X, **fit_kwargs)

        yt = check_endog(y, dtype=DTYPE, copy=False)
        Xt = X
        named_kwargs = self._get_kwargs(**fit_kwargs)

        # store original shape for later in-sample preds
        self.n_samples_ = yt.shape[0]

        for step_idx, name, transformer in self._iter(with_final=False):
            cloned_transformer = clone(transformer)
            kwargs = named_kwargs[name]
            yt, Xt = cloned_transformer.fit_transform(yt, Xt, **kwargs)

            # Replace the transformer of the step with the fitted
            # transformer.
            steps[step_idx] = (name, cloned_transformer)

        # Save the order of the columns so we can select in the predict phase
        self.x_feats_ = Xt.columns.tolist() \
            if isinstance(Xt, pd.DataFrame) else None

        # Now fit the final estimator
        kwargs = named_kwargs[steps[-1][0]]
        self._final_estimator.fit(yt, X=Xt, **kwargs)
        return self

    def _pre_predict(self, n_periods, X, **kwargs):
        """Runs transformation steps before predicting on data"""
        check_is_fitted(self, "steps_")

        # Push the arrays through the transformer stages, but ONLY the exog
        # transformer stages since we don't have a Y...
        Xt = X
        named_kwargs = self._get_kwargs(**kwargs)

        for step_idx, name, transformer in self._iter(with_final=False):
            if isinstance(transformer, BaseExogTransformer):
                kw = named_kwargs[name]

                # If it's a featurizer, we may also need to add 'n_periods'
                if isinstance(transformer, BaseExogFeaturizer):
                    num_p = kw.get("n_periods", None)
                    if num_p is not None and num_p != n_periods:
                        raise ValueError("Manually set 'n_periods' kwarg for "
                                         "step '%s' differs from forecasting "
                                         "n_periods (%r != %r)"
                                         % (name, num_p, n_periods))
                    kw["n_periods"] = n_periods

                # TODO: manual check to ensure Xt shape == n_periods shape?

                _, Xt = transformer.transform(y=None, X=Xt, **kw)

        # since some exog featurizers require exog input, and others don't,
        # feat orders may get wonky between fit & pred. Make sure we have them
        # in the same order when calling the model
        if self.x_feats_ is not None:
            Xt = Xt[self.x_feats_]

        # Now we should be able to run the prediction
        nm, est = self.steps_[-1]
        return Xt, est, named_kwargs[nm]

    def transform(self, n_periods=10, X=None, **kwargs):
        """Get the transformed X array

        Generate the X array ``n_periods`` in the future. This passes
        the provided exog variables through all transformation steps, returning
        it before it's passed into the model.

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
\
        **kwargs : keyword args
            Extra keyword arguments used for each stage's ``transform`` stage
            and the estimator's ``predict`` stage. Similar to scikit-learn
            pipeline keyword args, the keys are compound, comprised of the
            stage name and the argument name separated by a "__". For instance,
            if you have a FourierFeaturizer whose stage is named
            "fourier", your transform kwargs could resemble::

                {"fourier__n_periods": 50}

        Returns
        -------
        X_prime : pd.DataFrame
            The transformed exog array.
        """
        # Temporary shim until we remove `exogenous` support completely
        X, kwargs = pm_compat.get_X(X, **kwargs)

        n_periods = self._check_n_periods(n_periods, X)
        kwargs = _warn_for_deprecated(**kwargs)
        Xt, _, _ = self._pre_predict(n_periods, X, **kwargs)
        return Xt

    def predict_in_sample(self,
                          X=None,
                          start=None,
                          end=None,
                          dynamic=False,
                          return_conf_int=False,
                          alpha=0.05,
                          inverse_transform=True,
                          **kwargs):
        """Generate in-sample predictions from the fit pipeline.

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

        start : int, optional (default=None)
            Zero-indexed observation number at which to start forecasting, ie.,
            the first forecast is start.

        end : int, optional (default=None)
            Zero-indexed observation number at which to end forecasting, ie.,
            the first forecast is start.

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

        inverse_transform : bool, optional (default=True)
            Whether to inverse transform predictions, if they are in log or
            BoxCox scale. Any endog transformer will be inverse-transformed.

        **kwargs : keyword args
            Extra keyword arguments used for each stage's ``transform`` stage.
            Similar to scikit-learn pipeline keyword args, the keys are
            compound, comprised of the stage name and the argument name
            separated by a "__". For instance, if you have a FourierFeaturizer
            whose stage is named "fourier", your transform kwargs could
            resemble::

                {"fourier__n_periods": 50}

        Returns
        -------
        preds : array
            The predicted values.

        conf_int : array-like, shape=(n_periods, 2), optional
            The confidence intervals for the predictions. Only returned if
            ``return_conf_int`` is True.
        """
        # Temporary shim until we remove `exogenous` support completely
        X, kwargs = pm_compat.get_X(X, **kwargs)

        kwargs = _warn_for_deprecated(**kwargs)
        Xt, est, predict_kwargs = self._pre_predict(0, X, **kwargs)

        return_vals = est.predict_in_sample(
            X=Xt,
            start=start,
            end=end,
            return_conf_int=return_conf_int,
            alpha=alpha,
            dynamic=dynamic,
            **predict_kwargs)

        return self._post_predict(
            Xt, return_vals, return_conf_int, inverse_transform)

    def predict(self,
                n_periods=10,
                X=None,
                return_conf_int=False,
                alpha=0.05,
                inverse_transform=True,
                **kwargs):
        """Forecast future (transformed) values

        Generate predictions (forecasts) ``n_periods`` in the future.
        Note that if an ``X`` array was used in the model fit, it
        will be expected for the predict procedure and will fail otherwise.
        Forecasts may be transformed by the endogenous steps along the way and
        might be on a different scale than raw training/test data.

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

        inverse_transform : bool, optional (default=True)
            Whether to inverse transform predictions, if they are in log or
            BoxCox scale. Any endog transformer will be inverse-transformed.

        **kwargs : keyword args
            Extra keyword arguments used for each stage's ``transform`` stage
            and the estimator's ``predict`` stage. Similar to scikit-learn
            pipeline keyword args, the keys are compound, comprised of the
            stage name and the argument name separated by a "__". For instance,
            if you have a FourierFeaturizer whose stage is named
            "fourier", your transform kwargs could resemble::

                {"fourier__n_periods": 50}

        Returns
        -------
        forecasts : array-like, shape=(n_periods,)
            The array of transformed, forecasted values.

        conf_int : array-like, shape=(n_periods, 2), optional
            The confidence intervals for the forecasts. Only returned if
            ``return_conf_int`` is True.
        """
        # Temporary shim until we remove `exogenous` support completely
        X, kwargs = pm_compat.get_X(X, **kwargs)

        n_periods = self._check_n_periods(n_periods, X)
        kwargs = _warn_for_deprecated(**kwargs)
        Xt, est, predict_kwargs = self._pre_predict(
            n_periods, X, **kwargs)

        return_vals = est.predict(
            n_periods=n_periods,
            X=Xt,
            return_conf_int=return_conf_int,
            alpha=alpha,
            **predict_kwargs)

        return self._post_predict(
            Xt, return_vals, return_conf_int, inverse_transform)

    def _post_predict(self,
                      Xt,
                      return_vals,
                      return_conf_int,
                      inverse_transform):
        """Inverse-transform predictions to original data scale"""

        if not inverse_transform:
            return return_vals

        y_pred = return_vals
        conf_ints = None
        if return_conf_int:
            y_pred, conf_ints = y_pred

        # step through transformers in the reverse order
        for name, transformer in self.steps_[::-1]:
            if isinstance(transformer, BaseEndogTransformer):
                y_pred, Xt = transformer.inverse_transform(y_pred, Xt)
                if return_conf_int:
                    # inverse transform of Xt is irrelevant to y
                    # so only transform it once
                    conf_ints[:, 0], _ = transformer.inverse_transform(
                        conf_ints[:, 0], Xt)
                    conf_ints[:, 1], _ = transformer.inverse_transform(
                        conf_ints[:, 1], Xt)

        if return_conf_int:
            return y_pred, conf_ints
        return y_pred

    @if_delegate_has_method('_final_estimator')
    def summary(self):
        """Get a summary of the ARIMA model"""
        return self._final_estimator.summary()

    def update(self, y, X=None, maxiter=None, **kwargs):
        """Update an ARIMA or auto-ARIMA as well as any necessary transformers

        Passes the newly observed values through the appropriate endog
        transformations, and the ``X`` array through the exog transformers
        (updating where necessary) before finally updating the ARIMA model.

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
            Extra keyword arguments used for each stage's ``update`` stage.
            Similar to scikit-learn pipeline keyword args, the keys are
            compound, comprised of the stage name and the argument name
            separated by a "__".
        """
        check_is_fitted(self, "steps_")

        # Temporary shim until we remove `exogenous` support completely
        X, kwargs = pm_compat.get_X(X, **kwargs)

        # Push the arrays through all of the transformer steps that have the
        # appropriate update_and_transform method
        yt = y
        Xt = X
        named_kwargs = self._get_kwargs(**kwargs)

        for step_idx, name, transformer in self._iter(with_final=False):
            kw = named_kwargs[name]
            if hasattr(transformer, "update_and_transform"):
                yt, Xt = transformer.update_and_transform(
                    y=yt, X=Xt, **kw)
            else:
                yt, Xt = transformer.transform(yt, X=Xt, **kw)

        if self.x_feats_ is not None:
            Xt = Xt[self.x_feats_]

        # Now we can update the arima
        nm, est = self.steps_[-1]
        return est.update(
            yt, X=Xt, maxiter=maxiter, **named_kwargs[nm])
