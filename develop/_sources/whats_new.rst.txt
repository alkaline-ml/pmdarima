.. _whats_new:

======================
What's new in pmdarima
======================

As new releases of pmdarima are pushed out, the following list (introduced in
v0.8.1) will document the latest features.

`v2.0.5 <https://alkaline-ml.com/pmdarima/2.0.5>`_
-------------------------------------------------

* Remove support for Python 3.7 (end-of-life 2023-06-23)
* Remove support for Python 3.8 (end-of-life 2024-10-07)

`v2.0.4 <https://alkaline-ml.com/pmdarima/2.0.4>`_
-------------------------------------------------

* Pin numpy to ``<2.0.0`` while work is done to support numpy ``>=2.0.0``

`v2.0.3 <https://alkaline-ml.com/pmdarima/2.0.3>`_
-------------------------------------------------

* Fix a bug when generating permutations using ``numpy>=1.24.0``. See `#534 <https://github.com/alkaline-ml/pmdarima/pull/534>`_
* Fix a bug in ``transform`` when using ``scikit-learn>=1.2.0``, addressing `#539 <https://github.com/alkaline-ml/pmdarima/issues/539>`_

`v2.0.2 <http://alkaline-ml.com/pmdarima/2.0.2>`_
-------------------------------------------------

* Add support for Python 3.11
* Change minimum numpy version to ``1.21.2``

`v2.0.1 <http://alkaline-ml.com/pmdarima/2.0.1>`_
-------------------------------------------------

* Add support for macOS with M1 chip

`v2.0.0 <http://alkaline-ml.com/pmdarima/2.0.0>`_
-------------------------------------------------

* Potentially breaking changes:

  - Use of the ``exogenous`` keyword (deprecated in 1.8.0) will now raise a ``TypeError``

  - Use of the ``sarimax_kwargs`` keyword (deprecated in 1.5.1) will now raise a ``TypeError``

  - A falsey value for ARIMA's ``method`` argument (deprecated pre-1.5.0) will now raise a ``ValueError``

  - A falsey value for ARIMA's ``maxiter`` argument will now raise a ``ValueError`` (warning since 1.5.0)

  - ``pmdarima`` is no longer built for 32-bit architectures

  - macOS images are built using macOS 11 instead of macOS 10.15

* Bump numpy dependency to >= 1.21

* Expose ``fittedvalues`` in the public API. See `#493 <https://github.com/alkaline-ml/pmdarima/issues/493>`_

* Add support for ARM64 architecture. See `#434 <https://github.com/alkaline-ml/pmdarima/issues/434>`_

* Introduce new arg, ``preserve_series``, to ``pmdarima.utils.check_endog`` that will preserve or squeeze
  a Pandas ``Series`` object to preserve index information.

* Update Cython pinned version to include ``!=0.29.31``

`v1.8.5 <http://alkaline-ml.com/pmdarima/1.8.5>`_
-------------------------------------------------

* Add support for Python 3.10

* Remove support for Python 3.6


`v1.8.4 <http://alkaline-ml.com/pmdarima/1.8.4/>`_
--------------------------------------------------

* Add compatibility for ``statsmodels`` 0.13 and higher


`v1.8.3 <http://alkaline-ml.com/pmdarima/1.8.3/>`_
--------------------------------------------------

* Fix a bug in ``tsdisplay`` where a value of ``lag_max`` larger than the length of the series would create a cryptic numpy
  broadcasting error. This precondition will still cause an error, but now it is one the user can better understand.
  See `#440 <https://github.com/alkaline-ml/pmdarima/issues/440>`_

* Change ``numpy`` pin to ``numpy>=1.19.3`` (and build on lowest supported version) to no longer limit users' NumPy
  versions. This addresses `#449 <https://github.com/alkaline-ml/pmdarima/issues/449>`_

* Fix a bug where ``scikit-learn`` version ``1.0.0`` was raising ``ValueError`` when calling ``if_delegate_has_method``,
  addressing `#454 <https://github.com/alkaline-ml/pmdarima/issues/454>`_


`v1.8.2 <http://alkaline-ml.com/pmdarima/1.8.2/>`_
--------------------------------------------------

* Change ``numpy`` pin to ``~=1.19.0`` to avoid incompatibility issues, addressing `#423 <https://github.com/alkaline-ml/pmdarima/issues/423>`_


`v1.8.1 <http://alkaline-ml.com/pmdarima/1.8.1/>`_
--------------------------------------------------

* Address `issue 370 <https://github.com/alkaline-ml/pmdarima/issues/370>`_ where
  iterables were not accepted in the ARIMA order.

* Address `issue 407 <https://github.com/alkaline-ml/pmdarima/issues/407>`_ where
  the ``LogEndogTransformer`` could not be cloned in a pipeline.

* No longer pin Cython to ``<0.29.18``

* Add support for Python 3.9


`v1.8.0 <http://alkaline-ml.com/pmdarima/1.8.0/>`_
--------------------------------------------------

* Wheels are no longer built for ``pmdarima`` on Python <3.6, and backward-compatibility
  is no longer guaranteed for older python versions.

* The ``exogenous`` argument has been deprecated in favor of ``X`` - See `the RFC <https://github.com/alkaline-ml/pmdarima/pull/372>`_ and
  `the PR <https://github.com/alkaline-ml/pmdarima/pull/385>`_ for more information. Beginning in version 2.0,
  the ``exogenous`` argument will raise an error.

* Migrate random searches into the auto-solvers interface

* Random searches now perform unit root tests to prevent models with near non-invertible parameters

* The default value of ``suppress_warnings`` has changed to ``True``. The primary reason for this is that
  most warnings emitted come from unit root tests, which are very noisy. ``DeprecationWarnings`` and other
  warnings generated from user input will still be emitted.

* Move ``ModelFitWarning`` from ``pmdarima.arima.warnings`` to ``pmdarima.warnings``

* Fix a bug where the :class:`pmdarima.model_selection.RollingForecastCV` could produce
  too few splits for the given input data.

* Change pin for ``setuptools`` from ``<50.0.0`` to ``!=50.0.0``, addressing
  `#401 <https://github.com/alkaline-ml/pmdarima/issues/401>`_

* Change pin for ``statsmodels`` from ``<0.12.0`` to ``!=0.12.0``, addressing
  `#376 <https://github.com/alkaline-ml/pmdarima/issues/376>`_


`v1.7.1 <http://alkaline-ml.com/pmdarima/1.7.1/>`_
--------------------------------------------------

* Pin ``setuptools<50.0.0``

* Pin ``statsmodels<0.12``

* Warn for Python versions <3.6. We will remove Python 3.5 support in version 1.8.0


`v1.7.0 <http://alkaline-ml.com/pmdarima/1.7.0/>`_
--------------------------------------------------

* Address issue `#341 <https://github.com/alkaline-ml/pmdarima/issues/341>`_ where
  a combination of a large ``m`` value and ``D`` value could difference an array into
  being too small to test stationarity in the ADF test

* Fix issue `#351 <https://github.com/alkaline-ml/pmdarima/issues/351>`_ where a large
  value of ``m`` could prevent the seasonality test from completing.

* Fix issue `#354 <https://github.com/alkaline-ml/pmdarima/issues/354>`_ where models with
  near non-invertible roots could still be considered as candidate best-fits.

* Remove legacy pickling behavior that separates the statsmodels object from the pmdarima
  object. This breaks backwards compatibility with versions pre-1.2.0.

* Change default ``with_intercept`` in :func:`pmdarima.arima.auto_arima` to ``'auto'`` rather than
  ``True``. This will behave much like the current behavior, where a truthiness check will still return
  True, but allows the stepwise search to selectively change it to ``False`` in the presence of certain
  differencing conditions.

* Inverse endog transformation is now supported when ``return_conf_int=True`` on pipeline predictions
  (thanks to `skyetim <https://github.com/skyetim>`_)

* Fix a bug where the :class:`pmdarima.model_selection.SlidingWindowForecastCV` could produce
  too few splits for the given input data.

* Permit custom scoring metrics to be passed for out-of-sample scoring, as requested in
  `#368 <https://github.com/alkaline-ml/pmdarima/issues/368>`_.


`v1.6.1 <http://alkaline-ml.com/pmdarima/1.6.1/>`_
--------------------------------------------------

* Pin Cython to be ``>=0.29,<0.29.18``

* Pin statsmodels to be ``>=0.11``


`v1.6.0 <http://alkaline-ml.com/pmdarima/1.6.0/>`_
--------------------------------------------------

* Support newest versions of matplotlib

* Add new level of ``auto_arima`` error actions: "trace" which will warn for errors while dumping
  the original stacktrace.

* New featurizer: :class:`pmdarima.preprocessing.DateFeaturizer`. This can be used to create dummy
  and ordinal exogenous features and is useful when modeling pseudo-seasonal trends or time series
  with holes in them.

* Removes first-party conda distributions (see `#326 <https://github.com/alkaline-ml/pmdarima/issues/326>`_)

* Raise a ``ValueError`` in ``arima.predict_in_sample`` when ``start < d``


`v1.5.3 <http://alkaline-ml.com/pmdarima/1.5.3/>`_
--------------------------------------------------

* Adds first-party conda distributions as requested in `#173 <https://github.com/alkaline-ml/pmdarima/issues/173>`_

  - Due to dependency limitations, we only support 64-bit architectures and Python 3.6 or 3.7

* Adds Python 3.8 support as requested in `#199 <https://github.com/alkaline-ml/pmdarima/issues/199>`_

* Added :func:`pmdarima.datasets.load_gasoline` dataset

* Added integer levels of verbosity in the ``trace`` argument

* Added support for statsmodels 0.11+

* Added :func:`pmdarima.model_selection.cross_val_predict`, as requested in
  `#291 <https://github.com/alkaline-ml/pmdarima/issues/291>`_


`v1.5.2 <http://alkaline-ml.com/pmdarima/1.5.2/>`_
--------------------------------------------------

* Added ``pmdarima.show_versions`` as a utility for issue filing

* Fixed deprecation for ``check_is_fitted`` in newer versions of scikit-learn

* Adds the :func:`pmdarima.datasets.load_sunspots` method with R's `sunspots <https://www.rdocumentation.org/packages/datasets/versions/3.6.1/topics/sunspots>`_ dataset

* Adds the :func:`pmdarima.model_selection.train_test_split` method

* Fix bug where 1.5.1 documentation was labeled version "0.0.0".

* Fix bug reported in `#271 <https://github.com/alkaline-ml/pmdarima/issues/271>`_, where
  the use of ``threading.local`` to store stepwise context information may have broken
  job schedulers.

* Fix bug reported in `#272 <https://github.com/alkaline-ml/pmdarima/issues/272>`_, where
  the new default value of ``max_order`` can cause a ``ValueError`` even in default cases
  when ``stepwise=False``.


`v1.5.1 <http://alkaline-ml.com/pmdarima/1.5.1/>`_
--------------------------------------------------

* No longer use statsmodels' ``ARIMA`` or ``ARMA`` class under the hood; only use
  the ``SARIMAX`` model, which cuts back on a lot of errors/warnings we saw in the past.
  (`#211 <https://github.com/alkaline-ml/pmdarima/issues/211>`_)

* Defaults in the ``ARIMA`` class that have changed as a result of #211:

  - ``maxiter`` is now 50 (was ``None``)
  - ``method`` is now 'lbfgs' (was ``None``)
  - ``seasonal_order`` is now ``(0, 0, 0, 0)`` (was ``None``)
  - ``max_order`` is now 5 (was 10) and is no longer used as a constraint when ``stepwise=True``

* Correct bug where ``aicc`` always added 1 (for constant) to degrees of freedom,
  even when ``df_model`` accounted for the constant term.

* New :class:`pmdarima.arima.auto.StepwiseContext` feature for more control over
  fit duration (introduced by `@kpsunkara <https://github.com/kpsunkara>`_ in `#221 <https://github.com/alkaline-ml/pmdarima/pull/221>`_).

* Adds the :class:`pmdarima.preprocessing.LogEndogTransformer` class as discussed in
  `#205 <https://github.com/alkaline-ml/pmdarima/issues/205>`_

* Exogenous arrays are no longer cast to numpy array by default, and will pass pandas
  frames through to the model. This keeps variable names intact in the summary (`#222 <https://github.com/alkaline-ml/pmdarima/issues/222>`_)

* Added the ``prefix`` param to exogenous featurizers to allow the addition of meaningful
  names to engineered features.

* Added polyroot test of near non-invertibility when ``stepwise=True``. For
  models that are near non-invertible will be deprioritized in model selection
  as requested in `#208 <https://github.com/alkaline-ml/pmdarima/issues/208>`_.

* Removes ``pmdarima.arima.ARIMA.add_new_samples``, which was previously deprecated.
  Use :func:`pmdarima.arima.ARIMA.update` instead.

* The following args have been deprecated from the :class:`pmdarima.arima.ARIMA` class
  as well as :func:`pmdarima.arima.auto_arima` and any other calling methods/classes:

  - ``disp``:sup:`[1]`
  - ``callback``:sup:`[1]`
  - ``transparams``
  - ``solver``
  - ``typ``

  [1] These can still be passed to the ``fit`` method via ``**fit_kwargs``, but should
  no longer be passed to the model constructor.

* Added `diff_inv` function that is in parity with R's implementation,
  `diffinv <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/diffinv.html>`_,
  as requested in `#180 <https://github.com/alkaline-ml/pmdarima/issues/180>`_.

* Added `decompose` function that is in parity with R's implementation,
  `decompose <https://www.rdocumentation.org/packages/stats/versions/3.6.1/topics/decompose>`_,
  as requested in `#190 <https://github.com/alkaline-ml/pmdarima/issues/190>`_

`v1.4.0 <http://alkaline-ml.com/pmdarima/1.4.0/>`_
--------------------------------------------------

* Fixes `#191 <https://github.com/alkaline-ml/pmdarima/issues/191>`_, an issue where
  the OCSB test could raise ``ValueError: negative dimensions are not allowed" in OCSB test``

* Add option to automatically inverse-transform endogenous transformations when predicting
  from pipelines (`#197 <https://github.com/alkaline-ml/pmdarima/issues/197>`_)

* Add ``predict_in_sample`` to pipeline (`#196 <https://github.com/alkaline-ml/pmdarima/issues/196>`_)

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

* Fixes a bug in ``predict_in_sample``, as addressed in `#140 <https://github.com/alkaline-ml/pmdarima/issues/140>`_.

* Numpy 1.16+ is now required

* Statsmodels 0.10.0+ is now required

* Added ``sarimax_kwargs`` to ``ARIMA`` constructor and ``auto_arima`` function.
  This fixes `#146 <https://github.com/alkaline-ml/pmdarima/issues/146>`_


`v1.2.1 <http://alkaline-ml.com/pmdarima/1.2.1/>`_
--------------------------------------------------

* Pins scipy at 1.2.0 to avoid a statsmodels bug.


`v1.2.0 <http://alkaline-ml.com/pmdarima/1.2.0/>`_
--------------------------------------------------

* Adds the ``OCSBTest`` of seasonality, as discussed in `#88 <https://github.com/alkaline-ml/pmdarima/issues/88>`_

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

v1.1.1 is a patch release in response to `#104 <https://github.com/alkaline-ml/pmdarima/issues/104>`_

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

* Adds ``ARIMA.plot_diagnostics`` method, as requested in `#49 <https://github.com/alkaline-ml/pmdarima/issues/49>`_

* Adds new arg to ``ARIMA`` constructor and ``auto_arima``: ``with_intercept`` (default is True).

* New default for ``trend`` is no longer ``'c'``, it is ``None``.

* Adds ``to_dict`` method to ``ARIMA`` class to address `Issue #54 <https://github.com/alkaline-ml/pmdarima/issues/54>`_

* ARIMA serialization no longer stores statsmodels results wrappers in the cache,
  but bundles them into the pickle file. This solves `Issue #48 <https://github.com/alkaline-ml/pmdarima/issues/48>`_
  and only works on statsmodels 0.9.0+ since they've fixed a bug on their end.

* The ``'PMDARIMA_CACHE'`` and ``'PMDARIMA_CACHE_WARN_SIZE'`` environment variables are
  now deprecated, since they no longer need to be used.

* Added versioned documentation. All releases' doc (from 0.9.0 onward) is now available
  at ``alkaline-ml.com/pmdarima/<version>``

* Fixes bug in ``ADFTest`` where ``OLS`` was computed with ``method="pinv"`` rather
  than ``"method=qr"``. This fix means better parity with R's results. See
  `#71 <https://github.com/alkaline-ml/pmdarima/pull/71>`_ for more context.

* ``CHTest`` now solves linear regression with ``normalize=True``. This solves
  `#74 <https://github.com/alkaline-ml/pmdarima/issues/74>`_

* Python 3.7 is now supported(!!)


`v1.0.0 <http://alkaline-ml.com/pmdarima/1.0.0/>`_
--------------------------------------------------

* **Wheels are no longer built for Python versions < 3.5.** You may still be able to build
  from source, but support for 2.x python versions will diminish in future versions.

* Migrates namespace from 'pyramid-arima' to 'pmdarima'. This is due to the fact that
  a growing web-framework (also named Pyramid) is causing namespace collisions when
  both packages are installed on a machine. See `Issue #34 <https://github.com/alkaline-ml/pmdarima/issues/34>`_
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

* Fixes bug (`Issue #30 <https://github.com/alkaline-ml/pmdarima/issues/30>`_) in ``ARIMA``
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
