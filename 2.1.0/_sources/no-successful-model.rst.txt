.. _no_successful_model:

===================================
When no viable models can be found
===================================

For certain time series, the search may return no viable models::

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
        "Could not successfully fit a viable ARIMA model "
    ValueError: Could not successfully fit a viable ARIMA model to input data.
    See http://alkaline-ml.com/pmdarima/no-successful-model.html for more information on why this can happen.


This can happen for a number of reasons:

* Most commonly, the roots of your model may be nearly non-invertible, meaning the inverted roots
  lie too close to the unit circle. Here's a good `blog post <https://robjhyndman.com/hyndsight/arma-roots/>`_
  on the subject. Make sure ``trace`` is truthy in order to see these warnings when fitting your model.

* Sometimes, your data may not be stationary and can raise errors from statsmodels when fitting. In this case,
  the stepwise algorithm will filter out problem model fits. This can arise in a number of situations, ranging
  from non-stationarity to actual code errors. Setting ``error_action='trace'`` will log the stacktraces of
  any errors encountered during the search.

* Your input data may not be suitable for ARIMA modeling. For instance, it could be a simple polynomial
  or solved by linear regression (i.e., differencing the time series has made it perfectly constant).

Make sure to set ``trace`` to at least 1 in order to see the search progress, and to a value >1 to see the
maximum trace logging available. If you still cannot diagnose why you are getting this error message, consider
:ref:`filing_bugs`.
