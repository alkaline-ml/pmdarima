.. _whats_new:

======================
What's new in pmdarima
======================

As new releases of pmdarima are pushed out, the following list (introduced in
v0.8.1) will document the latest features.


`v1.5.0 <http://alkaline-ml.com/pmdarima/1.5.0/>`_
--------------------------------------------------

* No longer use statsmodels' ``ARIMA`` or ``ARMA`` class under the hood; only use
  the ``SARIMAX`` model, which cuts back on a lot of errors/warnings we saw in the past.
  (`#211 <https://github.com/tgsmith61591/pmdarima/issues/211>`_)

* Defaults in the ``ARIMA`` class that have changed as a result:

  - ``maxiter`` is now 50 (was ``None``)
  - ``method`` is now 'lbfgs' (was ``None``)
  - ``seasonal_order`` is now ``(0, 0, 0, 0)`` (was ``None``)

* Correct bug where ``aicc`` always added 1 (for constant) to degrees of freedom,
  even when ``df_model`` accounted for the constant term.


`v1.4.0 <http://alkaline-ml.com/pmdarima/1.4.0/>`_
--------------------------------------------------

* Fixes `#191 <https://github.com/tgsmith61591/pmdarima/issues/191>`_, an issue where
  the OCSB test could raise ``ValueError: negative dimensions are not allowed" in OCSB test``

* Add option to automatically inverse-transform endogenous transformations when predicting
  from pipelines (`#197 <https://github.com/tgsmith61591/pmdarima/issues/197>`_)

* Add ``predict_in_sample`` to pipeline (`#196 <https://github.com/tgsmith61591/pmdarima/issues/196>`_)

* Parameterize ``dtype`` option in datasets module

* Adds the ``model_selection`` submodule, which defines several different cross-validation
  classes as well as CV functions:

  - :class:`pmdarima.model_selection.RollingForecastCV`
  - :class:`pmdarima.model_selection.SlidingWindowForecastCV`
  - :func:`pmdarima.model_selection.cross_validate`
  - :func:`pmdarima.model_selection.cross_val_score`

* Adds the :func:`pmdarima.datasets.load_taylor` dataset


`v1.3.0 <http://alkaline-ml.com/pmdarima/1.3.0/>`_
--------------------------------------------------

* Adds a new dataset for stock prediction, along with an associated example (``load_msft``)

* Fixes a bug in ``predict_in_sample``, as addressed in `#140 <https://github.com/tgsmith61591/pmdarima/issues/140>`_.

* Numpy 1.16+ is now required

* Statsmodels 0.10.0+ is now required

* Added ``sarimax_kwargs`` to ``ARIMA`` constructor and ``auto_arima`` function.
  This fixes `#146 <https://github.com/tgsmith61591/pmdarima/issues/146>`_


`v1.2.1 <http://alkaline-ml.com/pmdarima/1.2.1/>`_
--------------------------------------------------

* Pins scipy at 1.2.0 to avoid a statsmodels bug.


`v1.2.0 <http://alkaline-ml.com/pmdarima/1.2.0/>`_
--------------------------------------------------

* Adds the ``OCSBTest`` of seasonality, as discussed in `#88 <https://github.com/tgsmith61591/pmdarima/issues/88>`_

* Default value of ``seasonal_test`` changes from "ch" to "ocsb" in ``auto_arima``

* Default value of ``test`` changes from "ch" to "ocsb" in ``nsdiffs``

* Adds benchmarking notebook and capabilities in ``pytest`` plugins

* Removes the following environment variables, which are now deprecated:
    * ``PMDARIMA_CACHE`` and ``PYRAMID_ARIMA_CACHE``
    * ``PMDARIMA_CACHE_WARN_SIZE`` and ``PYRAMID_ARIMA_CACHE_WARN_SIZE``
    * ``PYRAMID_MPL_DEBUG``
    * ``PYRAMID_MPL_BACKEND``

* Deprecates the ``is_stationary`` method in tests of stationarity. This will be removed in
  v1.4.0. Use ``should_diff`` instead.

* Adds two new datasets: ``airpassengers`` & ``austres``

* When using ``out_of_sample``, the out-of-sample predictions are now stored
  under the ``oob_preds_`` attribute.

* Adds a number of transformer classes including:
    * ``BoxCoxEndogTransformer``
    * ``FourierFeaturizer``

* Adds a ``Pipeline`` class resembling that of scikit-learn's, which allows the
  stacking of transformers together.

* Adds a class wrapper for ``auto_arima``: ``AutoARIMA``. This is allows auto-ARIMA
  to be used with pipelines.


`v1.1.1 <http://alkaline-ml.com/pmdarima/1.1.1/>`_
--------------------------------------------------

v1.1.1 is a patch release in response to `#104 <https://github.com/tgsmith61591/pmdarima/issues/104>`_

* Deprecates the ``ARIMA.add_new_observations`` method. This method originally was designed to support
  updating the endogenous/exogenous arrays with new observations without changing the model parameters,
  but achieving this behavior for each of statsmodels' ``ARMA``, ``ARIMA`` and ``SARIMAX`` classes proved
  nearly impossible, given the extremely complex internals of statmodels estimators.

* Replaces ``ARIMA.add_new_observations`` with ``ARIMA.update``. This allows the user to update the model
  with new observations by taking ``maxiter`` new steps from the existing model coefficients and allowing the MLE to
  converge to an updated set of model parameters.

* Changes default ``maxiter`` to None, using 50 for seasonal models and 500 for non-seasonal models (as
  statsmodels does). The default value used to be 50 for all models.

* New behavior in ``ARIMA.fit`` allows ``start_params`` and ``maxiter`` to be passed as ``**fit_args``,
  overriding the use of their corresponding instance attributes.


`v1.1.0 <http://alkaline-ml.com/pmdarima/1.1.0/>`_
--------------------------------------------------

* Adds ``ARIMA.plot_diagnostics`` method, as requested in `#49 <https://github.com/tgsmith61591/pmdarima/issues/49>`_

* Adds new arg to ``ARIMA`` constructor and ``auto_arima``: ``with_intercept`` (default is True).

* New default for ``trend`` is no longer ``'c'``, it is ``None``.

* Adds ``to_dict`` method to ``ARIMA`` class to address `Issue #54 <https://github.com/tgsmith61591/pmdarima/issues/54>`_

* ARIMA serialization no longer stores statsmodels results wrappers in the cache,
  but bundles them into the pickle file. This solves `Issue #48 <https://github.com/tgsmith61591/pmdarima/issues/48>`_
  and only works on statsmodels 0.9.0+ since they've fixed a bug on their end.

* The ``'PMDARIMA_CACHE'`` and ``'PMDARIMA_CACHE_WARN_SIZE'`` environment variables are
  now deprecated, since they no longer need to be used.

* Added versioned documentation. All releases' doc (from 0.9.0 onward) is now available
  at ``alkaline-ml.com/pmdarima/<version>``

* Fixes bug in ``ADFTest`` where ``OLS`` was computed with ``method="pinv"`` rather
  than ``"method=qr"``. This fix means better parity with R's results. See
  `#71 <https://github.com/tgsmith61591/pmdarima/pull/71>`_ for more context.

* ``CHTest`` now solves linear regression with ``normalize=True``. This solves
  `#74 <https://github.com/tgsmith61591/pmdarima/issues/74>`_

* Python 3.7 is now supported(!!)


`v1.0.0 <http://alkaline-ml.com/pmdarima/1.0.0/>`_
--------------------------------------------------

* **Wheels are no longer built for Python versions < 3.5.** You may still be able to build
  from source, but support for 2.x python versions will diminish in future versions.

* Migrates namespace from 'pyramid-arima' to 'pmdarima'. This is due to the fact that
  a growing web-framework (also named Pyramid) is causing namespace collisions when
  both packages are installed on a machine. See `Issue #34 <https://github.com/tgsmith61591/pmdarima/issues/34>`_
  for more detail.

* Removes redundant Travis tests

* Automates documentation build on Circle CI

* Moves lots of the build/test functionality into the ``Makefile`` for ease.

* Warns for impending deprecation of various environment variable name changes. The following
  will be completely switched over in version 1.2.0:

  - ``'PYRAMID_MPL_DEBUG'`` will become ``'PMDARIMA_MPL_DEBUG'``
  - ``'PYRAMID_MPL_BACKEND'`` will become ``'PMDARIMA_MPL_BACKEND'``
  - ``'PYRAMID_ARIMA_CACHE_WARN_SIZE'`` will become ``'PMDARIMA_CACHE_WARN_SIZE'``


`v0.9.0 <http://alkaline-ml.com/pmdarima/0.9.0/>`_
--------------------------------------------------

* Explicitly catches case in ``auto_arima`` where a value of ``m`` that is too large may over-estimate
  ``D``, causing the time series to be differenced down to an empty array. This is now handled by
  raising a separate error for this case that better explains what happened.

* Re-pickling an ``ARIMA`` will no longer remove the location on disk of the cached ``statsmodels``
  ARIMA models. Older versions encountered an issue where an older version of the model would be
  reinstated and immediately fail due to an OSError since the cached state no longer existed. This
  means that a user must be very intentional about clearing out the pyramid cache over time.

* Adds pyramid cache check on initial import to warn user if the cache size has grown too large.

* If ``d`` or ``D`` are explicitly defined for ``auto_arima`` (rather than ``None``), do not
  raise an error if they exceed ``max_d`` or ``max_D``, respectively.

* Adds Circle CI for validating PyPy builds (rather than CPython)

* Deploys python wheel for version 3.6 on Linux and Windows

* Includes warning for upcoming package name change (``pmdarima``).

v0.8.1
------

* New ``ARIMA`` instance attributes

  - The ``pkg_version_`` attribute (assigned on model ``fit``) is new as of version 0.8.0.
    On unpickling, if the current Pyramid version does not match the version under which it
    was serialized, a ``UserWarning`` will be raised.

* Addition of the ``_config.py`` file at the top-level of the package

  - Specifies the location of the ARIMA result pickles (see :ref:`serializing`)
  - Specifies the ARIMA result pickle name pattern

* Fixes bug (`Issue #30 <https://github.com/tgsmith61591/pmdarima/issues/30>`_) in ``ARIMA``
  where using CV with differencing and no seasonality caused a dim mismatch in the model's
  exog array and its endog array

* New dataset: :ref:`woolyrnq` (from R's ``forecast`` package).

* Visualization utilities available at the top level of the package:

    - ``plot_acf``
    - ``plot_pacf``
    - ``autocorr_plot``

* Updates documentation with significantly more examples and API references.


v0.7.0
------

* ``out_of_sample_size`` behavior in :class:`pmdarima.arima.ARIMA`

  - In prior versions, the ``out_of_sample_size`` (OOSS) parameter misbehaved in the sense that it
    ended up fitting the model on the entire sample, and scoring the number specified. This
    behavior changed in v0.7.0. Going forward, when OOSS is not None,
    ARIMA models will be fit on :math:`n - OOSS` samples, scored on the last OOSS samples,
    and the held-out samples are then added to the model.

* Adds ``add_new_samples`` method to :class:`pmdarima.arima.ARIMA`

  - This method adds new samples to the model, effectively refreshing the point from
    which it creates new forecasts without impacting the model parameters.

* Adds confidence intervals on ``predict`` in :class:`pmdarima.arima.ARIMA`

  - When ``return_conf_int`` is true, the confidence intervals will now be returned
    with the forecasts.

v0.6.5
------

* :class:`pmdarima.arima.CHTest` of seasonality

  - No longer computes the :math:`U` or :math:`V` matrix in the SVD computation in the
    Canova-Hansen test. This makes the test *much* faster.
