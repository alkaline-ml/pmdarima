# -*- coding: utf-8 -*-

from itertools import islice

from sklearn.base import BaseEstimator, clone
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from .base import BaseARIMA
from .preprocessing.base import BaseTransformer
from .preprocessing.exog.base import BaseExogTransformer

__all__ = ['Pipeline']


class Pipeline(BaseEstimator):
    """A pipeline of transformers with an optional final estimator stage

    The pipeline object chains together an arbitrary number of named, ordered
    transformations, passing the output from one as the input to the next. As
    the last stage, an ``ARIMA`` or ``AutoARIMA`` object will be fit.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.

    This pipeline takes after the scikit-learn ``sklearn.Pipeline`` object,
    which behaves similarly but does not share the same time-series interface
    that ``pmdarima`` follows.

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
    ...     ("fourier", FourierFeaturizer(m=12)),
    ...     ("arima", AutoARIMA(seasonal=False, stepwise=True,
    ...                         suppress_warnings=True,
    ...                         error_action='ignore'))
    ... ])
    >>> pipeline.fit(wineind)
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

    def __getitem__(self, ind):
        """Returns a sub-pipeline or a single esimtator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """
        if isinstance(ind, slice):
            if ind.step not in (1, None):
                raise ValueError('Pipeline slicing only supports a step of 1')
            return self.__class__(self.steps[ind])
        try:
            name, est = self.steps[ind]
        except TypeError:
            # Not an int, try get step by name
            return self.named_steps[ind]
        return est

    @property
    def named_steps(self):
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps))

    @property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return estimator

    def fit(self, y, exogenous=None, **fit_kwargs):
        """Fit the pipeline of transformers and the ARIMA model

        Chain the time-series and exogenous arrays through a series of
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

        exogenous : array-like, shape=[n_obs, n_vars], optional (default=None)
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

        yt = y
        Xt = exogenous
        named_kwargs = self._get_kwargs(**fit_kwargs)

        for step_idx, name, transformer in self._iter(with_final=False):
            cloned_transformer = clone(transformer)
            kwargs = named_kwargs[name]
            yt, Xt = cloned_transformer.fit_transform(yt, Xt, **kwargs)

            # Replace the transformer of the step with the fitted
            # transformer.
            steps[step_idx] = (name, cloned_transformer)

        # Now fit the final estimator
        kwargs = named_kwargs[steps[-1][0]]
        self._final_estimator.fit(yt, exogenous=Xt, **kwargs)
        return self

    def predict(self, n_periods=10, exogenous=None,
                return_conf_int=False, alpha=0.05, **kwargs):
        """Forecast future values

        Generate predictions (forecasts) ``n_periods`` in the future.
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

        **kwargs : keyword args
            Extra keyword arguments used for each stage's ``transform`` stage
            and the estimator's ``predict`` stage. Similar to scikit-learn
            pipeline keyword args, the keys are compound, comprised of the
            stage name and the argument name separated by a "__". For instance,
            if you have a FourierFeaturizer whose stage is named
            "fourier", your transform kwargs could resemble::

                {"fourier__h": 50}

        Returns
        -------
        forecasts : array-like, shape=(n_periods,)
            The array of fore-casted values.

        conf_int : array-like, shape=(n_periods, 2), optional
            The confidence intervals for the forecasts. Only returned if
            ``return_conf_int`` is True.
        """
        check_is_fitted(self, "steps_")

        # Push the arrays through the transformer stages, but ONLY the exog
        # transformer stages since we don't have a Y...
        Xt = exogenous
        named_kwargs = self._get_kwargs(**kwargs)

        for step_idx, name, transformer in self._iter(with_final=False):
            if isinstance(transformer, BaseExogTransformer):
                kw = named_kwargs[name]
                _, Xt = transformer.transform(y=None, exogenous=Xt, **kw)

        # Now we should be able to run the prediction
        nm, est = self.steps_[-1]
        return est.predict(
            n_periods=n_periods, exogenous=Xt,
            return_conf_int=return_conf_int,
            alpha=alpha, **named_kwargs[nm])

    # todo: update
