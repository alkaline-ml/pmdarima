.. _seasonal_differencing_issues:

============================================
Encountering issues in seasonal differencing
============================================

For certain time series, the seasonal differencing operation may fail::

    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
        "Could not successfully fit a viable ARIMA model "
    There are no more samples after a first-order seasonal differencing. See
    http://alkaline-ml.com/pmdarima/seasonal-differencing-issues.html for a
    more in-depth explanation and potential work-arounds.


In short, the seasonal differencing test has detected your time series could benefit
from a non-zero seasonal differencing term, ``D``, but your data is exhausted after
differencing it by ``m``. Basically, your dataset is too small to be differenced by ``m``.
You only have several options as a work-around here:

* Use a larger training set.

* Determine whether or not you've set the appropriate ``m``. Should it be smaller? See
  :ref:`period` for more information on the topic.

* Manually set ``D=0`` in the :func:`pmdarima.arima.auto_arima` call. This is the least
  desirable solution, since it skips a step that could lead to a better model.

The best decision is always to use a larger training set, but sometimes that simply
is not possible. Make sure to set ``trace`` to at least 1 in order to see the search progress, and to a value >1 to see the
maximum trace logging available. If you still cannot diagnose why you are getting this error message, consider
:ref:`filing_bugs`.
