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
