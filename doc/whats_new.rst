.. _whats_new:

=====================
What's new in Pyramid
=====================

As new releases of Pyramid are pushed out, the following list (introduced in
v0.8.1) will document the latest features.

v0.8.1
------

* ``ARIMA`` instance attributes

  - The ``pkg_version_`` attribute (assigned on model ``fit``) is new as of version 0.8.0.
    On unpickling, if the current Pyramid version does not match the version under which it
    was serialized, a ``UserWarning`` will be raised.

* Addition of the ``_config.py`` file at the top-level of the package

  - Specifies the location of the ARIMA result pickles (see :ref:`serializing`)
  - Specifies the ARIMA result pickle name pattern

* Fix bug (`Issue #30 <https://github.com/tgsmith61591/pyramid/issues/30>`_) in ``ARIMA``
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

* ``ARIMA`` ``out_of_sample_size`` behavior

  - In prior versions, the ``out_of_sample_size`` (OOSS) parameter misbehaved in the sense that it
    ended up fitting the model on the entire sample, and scoring the number specified. This
    behavior changed in v0.7.0. Going forward, when OOSS is not None,
    ARIMA models will be fit on :math:`n - OOSS` samples, scored on the last OOSS samples,
    and the held-out samples are then added to the model.

* ``ARIMA`` ``add_new_samples`` method

  - This method adds new samples to the model, effectively refreshing the point from
    which it creates new forecasts without impacting the model parameters.

* ``ARIMA`` confidence intervals on ``predict``

  - When ``return_conf_int`` is true, the confidence intervals will now be returned
    with the forecasts.

v0.6.5
------

* ``CHTest`` of seasonality

  - No longer compute the :math:`U` or :math:`V` matrix in the SVD computation in the
    Canova-Hansen test. This makes the test *much* faster.
