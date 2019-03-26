.. _refreshing:

============================
Refreshing your ARIMA models
============================

There are two ways to keep your models up-to-date with pmdarima:

1. Periodically, your ARIMA will need to be refreshed given new observations. See
   `this discussion <https://stats.stackexchange.com/questions/34139/updating-arima-models-at-frequent-intervals>`_
   and `this one <https://stats.stackexchange.com/questions/57745/what-do-you-consider-a-new-model-versus-an-updated-model-time-series>`_
   on either re-using ``auto_arima``-estimated order terms or re-fitting altogether.

2. If you're not ready to totally refresh your model parameters, but would like to add observations to
   your model (so new forecasts originate from the latest samples) with minor parameter updates, the ARIMA class makes it
   possible to `add new samples <./modules/generated/pmdarima.arima.ARIMA.html#pmdarima.arima.ARIMA.update>`_.
   See `this example <auto_examples/arima/example_add_new_samples.html#adding-new-observations-to-your-model>`_
   for more info.


Updating your model with new observations
-----------------------------------------

The easiest way to keep your model up-to-date without completely refitting it is simply to
update your model with new observations so that future forecasts take the newest observations
into consideration. Assume that you fit the following model:

.. code-block:: python

    import pmdarima as pm
    from pmdarima.datasets import load_wineind

    y = load_wineind()
    train, test = y[:125], y[125:]

    # Fit an ARIMA
    arima = pm.ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
    arima.fit(y)

After fitting and persisting your model (see :ref:`serializing`), you use your model
to produce forecasts. After a few forecasts, you want to record the *actual* observed
values so your model considers them when making newer forecasts:

.. code-block:: python

    arima.update(test)  # pretend these are the new ones

Your model will now produce forecasts from the *new* latest observations. Of course,
you'll have to re-persist your ARIMA model after updating it! Internally, this step
uses the existing parameters, taking a small amount of steps and allowing MLE to
update your parameters a small amount. You can pass the ``maxiter`` to control the
amount your model updates.
