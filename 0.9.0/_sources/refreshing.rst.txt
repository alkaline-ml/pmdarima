.. _refreshing:

============================
Refreshing your ARIMA models
============================

There are two ways to keep your models up-to-date with Pyramid:

1. Periodically, your ARIMA will need to be refreshed given new observations. See
   `this discussion <https://stats.stackexchange.com/questions/34139/updating-arima-models-at-frequent-intervals>`_
   and `this one <https://stats.stackexchange.com/questions/57745/what-do-you-consider-a-new-model-versus-an-updated-model-time-series>`_
   on either re-using ``auto_arima``-estimated order terms or re-fitting altogether.

2. If you're not ready to refresh your model parameters, but would like to add observations to
   your model (so new forecasts originate from the latest samples), the ARIMA class makes it
   possible to `add new samples <./modules/generated/pyramid.arima.ARIMA.html#pyramid.arima.ARIMA.add_new_observations>`_.
   See `this example <auto_examples/arima/example_add_new_samples.html#adding-new-observations-to-your-model>`_
   for more info.


Adding observations to your model
---------------------------------

The easiest way to keep your model up-to-date without refreshing it is simply to
add observations to your model so that future forecasts take the newest observations
into consideration. Assume that you fit the following model:

.. code-block:: python

    import pyramid as pm
    from pyramid.datasets import load_wineind

    y = load_wineind()
    train, test = y[:125], y[125:]

    # Fit an ARIMA
    arima = pm.ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
    arima.fit(y)

After fitting and persisting your model (see :ref:`serializing`), you use your model
to produce forecasts. After a few forecasts, you want to record the *actual* observed
values so your model considers them when making newer forecasts:

.. code-block:: python

    arima.add_new_observations(test)  # pretend these are the new ones

Your model will now produce forecasts from the *new* latest observations. Of course,
you'll have to re-persist your ARIMA model after updating it!
