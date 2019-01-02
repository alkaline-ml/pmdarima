.. _api_ref:

=============
API Reference
=============

.. include:: ../includes/api_css.rst

This is the class and function reference for ``pmdarima``. Please refer to
the :ref:`full user guide <user_guide>` for further details, as the class and
function raw specifications may not be enough to give full guidelines on their
uses.


.. _arima_ref:

:mod:`pmdarima.arima`: ARIMA estimator & differencing tests
===========================================================

The ``pmdarima.arima`` sub-module defines the ``ARIMA`` estimator and the
``auto_arima`` function, as well as a set of tests of seasonality and
stationarity.

.. automodule:: pmdarima.arima
    :no-members:
    :no-inherited-members:

ARIMA estimator & statistical tests
-----------------------------------

**User guide:** See the :ref:`seasonality` and :ref:`enforcing_stationarity`
sections for further details.

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: class.rst

    arima.ADFTest
    arima.ARIMA
    arima.CHTest
    arima.KPSSTest
    arima.PPTest

ARIMA auto-parameter selection
------------------------------

**User guide:** See the :ref:`tips_and_tricks` section for further details.

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.auto_arima


Differencing helpers
--------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    arima.is_constant
    arima.ndiffs
    arima.nsdiffs


.. _datasets_ref:

:mod:`pmdarima.datasets`: Toy univariate timeseries datasets
============================================================

The ``pmdarima.datasets`` submodule provides several different univariate time-
series datasets used in various examples and tests across the package. If you
would like to prototype a model, this is a good place to find easy-to-access data.

**User guide:** See the :ref:`datasets` section for further details.

.. automodule:: pmdarima.datasets
    :no-members:
    :no-inherited-members:

Dataset loading functions
-------------------------

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    datasets.load_airpassengers
    datasets.load_austres
    datasets.load_heartrate
    datasets.load_lynx
    datasets.load_wineind
    datasets.load_woolyrnq


.. _utils_ref:

:mod:`pmdarima.utils`: Utilities
================================

Utilities and array differencing functions used commonly across the package.

.. automodule:: pmdarima.utils
    :no-members:
    :no-inherited-members:

Array helper functions & metaestimators
---------------------------------------

.. currentmodule:: pmdarima

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

.. currentmodule:: pmdarima

.. autosummary::
    :toctree: generated/
    :template: function.rst

    utils.autocorr_plot
    utils.plot_acf
    utils.plot_pacf
