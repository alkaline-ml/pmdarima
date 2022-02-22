.. _migration:

============================
``pmdarima`` Migration guide
============================

In `issue #34 <https://github.com/alkaline-ml/pmdarima/issues/34>`_ we made the
decision to migrate from the ``pyramid-arima`` namespace to the ``pmdarima``
namespace to avoid collisions with the web framework named ``pyramid``.

Migration is simple:

.. code-block:: bash

    $ pip install pmdarima

Rather that importing functions and modules from the ``pyramid`` package, simply
import from ``pmdarima`` instead:

.. code-block:: python

    from pmdarima.arima import auto_arima

Or just import it as a namespace:

.. code-block:: python

    import pmdarima as pm
    my_model = pm.auto_arima(my_timeseries)

For further installation instructions, check out the :ref:`setup` and :ref:`quickstart` guides.
