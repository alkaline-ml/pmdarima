.. _api_ref:

=============
API Reference
=============

This is the class and function reference for pyramid. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _arima_ref:

:mod:`pyramid.arima`: ARIMA estimator and associated tests
==========================================================

The ``pyramid.arima`` sub-module defines the ``ARIMA`` estimator and the
``auto_arima`` function, as well as a set of tests of seasonality and
stationarity.

.. automodule:: pyramid.arima
    :no-members:
    :no-inherited-members:

ARIMA estimator
---------------

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: class.rst

    arima.ARIMA

ARIMA auto-parameter selection
------------------------------

`auto_arima` selects the best model order for an ARIMA estimator by minimizing
one of several scoring criteria. This is typically the most effective way to
fit an ARIMA model unless you have apriori subject matter expertise regarding
the trends within your data.

**User guide:** See the :ref:`tips_and_tricks` section for further details.

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.auto_arima

Tests of seasonality
--------------------

Tests of seasonality are important if your timeseries data exhibits seasonal
trends (i.e., Black Friday trends in retail, dependencies on weather patterns,
etc.). Pyramid currently implements only one test of seasonality, but more will
be added in future versions.

**User guide:** See the :ref:`seasonality` section for further details.

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: class.rst

    arima.CHTest


Tests of stationarity
---------------------

Timeseries data must be stationary in order to model it well. Pyramid implements
several tests of stationarity.

**User guide:** See the :ref:`enforcing_stationarity` section for further details.

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: class.rst

    arima.ADFTest
    arima.KPSSTest
    arima.PPTest


Differencing helpers
--------------------

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.is_constant
    arima.ndiffs
    arima.nsdiffs


.. _datasets_ref:

:mod:`pyramid.datasets`: Toy univariate timeseries datasets
===========================================================

The ``pyramid.datasets`` submodule provides several different univariate time-
series datasets used in various examples and tests across the package. If you
would like to prototype a model, this is a good place to find easy-to-access data.

**User guide:** See the :ref:`datasets` section for further details.

.. automodule:: pyramid.datasets
    :no-members:
    :no-inherited-members:

Dataset loading functions
-------------------------

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: function.rst

    datasets.load_heartrate
    datasets.load_lynx
    datasets.load_wineind


.. _utils_ref:

:mod:`pyramid.utils`: Utilities
===============================

Utilities and array differencing functions used commonly across the package.

.. automodule:: pyramid.utils
    :no-members:
    :no-inherited-members:

Array helper functions & metaestimators
---------------------------------------

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.acf
    utils.as_series
    utils.c
    utils.diff
    utils.if_has_delegate
    utils.is_iterable
    utils.pacf

Plotting utilities & wrappers
-----------------------------

.. currentmodule:: pyramid

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.autocorr_plot
    utils.plot_acf
    utils.plot_pacf
