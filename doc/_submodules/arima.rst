=================
``pyramid.arima``
=================

The ``pyramid.arima`` sub-module defines the ``ARIMA`` estimator and the
``auto_arima`` function, as well as a set of tests of seasonality and
stationarity.

.. toctree::
    :caption: Tests of seasonality and stationarity
    :maxdepth: 2

    arima.seasonality <./arima.seasonality.rst>
    arima.stationarity <./arima.stationarity.rst>

auto_arima
----------
.. autofunction:: pyramid.arima.auto_arima

ndiffs
------
.. autofunction:: pyramid.arima.ndiffs

nsdiffs
-------
.. autofunction:: pyramid.arima.nsdiffs

ARIMA
-----
.. autoclass:: pyramid.arima.ARIMA
    :members:
    :undoc-members:
    :show-inheritance: