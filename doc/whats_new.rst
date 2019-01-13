.. _whats_new:

======================
What's new in pmdarima
======================

As new releases of pmdarima are pushed out, the following list (introduced in
v0.8.1) will document the latest features.

`v1.2.0 <http://alkaline-ml.com/pmdarima/1.2.0/>`_
--------------------------------------------------

* Added the ``OCSBTest`` of seasonality, as discussed in `#88 <https://github.com/tgsmith61591/pmdarima/issues/88>`_

* Default value of ``seasonal_test`` changed from "ch" to "ocsb" in ``auto_arima``

* Default value of ``test`` changed from "ch" to "ocsb" in ``nsdiffs``

* Add benchmarking notebook and capabilities in ``pytest`` plugins

* Remove the following environment variables, which are now deprecated:
    * ``'PMDARIMA_CACHE'`` and ``'PYRAMID_ARIMA_CACHE'``
    * ``'PMDARIMA_CACHE_WARN_SIZE'`` and ``PYRAMID_ARIMA_CACHE_WARN_SIZE``
    * ``PYRAMID_MPL_DEBUG``
    * ``PYRAMID_MPL_BACKEND``


`v1.1.0 <http://alkaline-ml.com/pmdarima/1.1.0/>`_
--------------------------------------------------

* Added ``ARIMA.plot_diagnostics`` method, as requested in `#49 <https://github.com/tgsmith61591/pmdarima/issues/49>`_

* Added new arg to ``ARIMA`` constructor and ``auto_arima``: ``with_intercept`` (default is True).

* New default for ``trend`` is no longer ``'c'``, it is ``None``.

* Added ``to_dict`` method to ``ARIMA`` class to address `Issue #54 <https://github.com/tgsmith61591/pmdarima/issues/54>`_

* ARIMA serialization no longer stores statsmodels results wrappers in the cache,
  but bundles them into the pickle file. This solves `Issue #48 <https://github.com/tgsmith61591/pmdarima/issues/48>`_
  and only works on statsmodels 0.9.0+ since they've fixed a bug on their end.

* The ``'PMDARIMA_CACHE'`` and ``'PMDARIMA_CACHE_WARN_SIZE'`` environment variables are
  now deprecated, since they no longer need to be used.

* Added versioned documentation. All releases' doc (from 0.9.0 onward) is now available
  at ``alkaline-ml.com/pmdarima/<version>``

* Fix bug in ``ADFTest`` where ``OLS`` was computed with ``method="pinv"`` rather
  than ``"method=qr"``. This fix means better parity with R's results. See
  `#71 <https://github.com/tgsmith61591/pmdarima/pull/71>`_ for more context.

* ``CHTest`` now solves linear regression with ``normalize=True``. This solves
  `#74 <https://github.com/tgsmith61591/pmdarima/issues/74>`_

* Python 3.7 is now supported(!!)


`v1.0.0 <http://alkaline-ml.com/pmdarima/1.0.0/>`_
--------------------------------------------------

* **Wheels will no longer be built for Python versions < 3.5.** You may still be able to build
  from source, but support for 2.x python versions will diminish in future versions.

* Migrate namespace from 'pyramid-arima' to 'pmdarima'. This is due to the fact that
  a growing web-framework (also named Pyramid) is causing namespace collisions when
  both packages are installed on a machine. See `Issue #34 <https://github.com/tgsmith61591/pmdarima/issues/34>`_
  for more detail.

* Remove redundant Travis tests

* Automate documentation build on Circle CI

* Move lots of the build/test functionality into the ``Makefile`` for ease.

* Warn for impending deprecation of various environment variable name changes. The following
  will be completely switched over in version 1.2.0:

  - ``'PYRAMID_MPL_DEBUG'`` will become ``'PMDARIMA_MPL_DEBUG'``
  - ``'PYRAMID_MPL_BACKEND'`` will become ``'PMDARIMA_MPL_BACKEND'``
  - ``'PYRAMID_ARIMA_CACHE_WARN_SIZE'`` will become ``'PMDARIMA_CACHE_WARN_SIZE'``


`v0.9.0 <http://alkaline-ml.com/pmdarima/0.9.0/>`_
--------------------------------------------------

* Explicitly catch case in ``auto_arima`` where a value of ``m`` that is too large may over-estimate
  ``D``, causing the time series to be differenced down to an empty array. This is now handled by
  raising a separate error for this case that better explains what happened.

* Re-pickling an ``ARIMA`` will no longer remove the location on disk of the cached ``statsmodels``
  ARIMA models. Older versions encountered an issue where an older version of the model would be
  reinstated and immediately fail due to an OSError since the cached state no longer existed. This
  means that a user must be very intentional about clearing out the pyramid cache over time.

* Added pyramid cache check on initial import to warn user if the cache size has grown too large.

* If ``d`` or ``D`` are explicitly defined for ``auto_arima`` (rather than ``None``), do not
  raise an error if they exceed ``max_d`` or ``max_D``, respectively.

* Added Circle CI for validating PyPy builds (rather than CPython)

* Deploy python wheel for version 3.6 on Linux and Windows

* Include warning for upcoming package name change (``pmdarima``).

v0.8.1
------

* ``ARIMA`` instance attributes

  - The ``pkg_version_`` attribute (assigned on model ``fit``) is new as of version 0.8.0.
    On unpickling, if the current Pyramid version does not match the version under which it
    was serialized, a ``UserWarning`` will be raised.

* Addition of the ``_config.py`` file at the top-level of the package

  - Specifies the location of the ARIMA result pickles (see :ref:`serializing`)
  - Specifies the ARIMA result pickle name pattern

* Fix bug (`Issue #30 <https://github.com/tgsmith61591/pmdarima/issues/30>`_) in ``ARIMA``
  where using CV with differencing and no seasonality caused a dim mismatch in the model's
  exog array and its endog array

* New dataset: :ref:`woolyrnq` (from R's ``forecast`` package).

* Visualization utilities available at the top level of the package:

    - ``plot_acf``
    - ``plot_pacf``
    - ``autocorr_plot``

* Updated documentation with significantly more examples and API references.


v0.7.0
------

* ``out_of_sample_size`` behavior in :class:`pmdarima.arima.ARIMA`

  - In prior versions, the ``out_of_sample_size`` (OOSS) parameter misbehaved in the sense that it
    ended up fitting the model on the entire sample, and scoring the number specified. This
    behavior changed in v0.7.0. Going forward, when OOSS is not None,
    ARIMA models will be fit on :math:`n - OOSS` samples, scored on the last OOSS samples,
    and the held-out samples are then added to the model.

* ``add_new_samples`` method added to :class:`pmdarima.arima.ARIMA`

  - This method adds new samples to the model, effectively refreshing the point from
    which it creates new forecasts without impacting the model parameters.

* Add confidence intervals on ``predict`` in :class:`pmdarima.arima.ARIMA`

  - When ``return_conf_int`` is true, the confidence intervals will now be returned
    with the forecasts.

v0.6.5
------

* :class:`pmdarima.arima.CHTest` of seasonality

  - No longer compute the :math:`U` or :math:`V` matrix in the SVD computation in the
    Canova-Hansen test. This makes the test *much* faster.
