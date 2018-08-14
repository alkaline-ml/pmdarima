.. _auto_arima:

======================
The auto_arima process
======================

Fit the best `ARIMA` model to a univariate time series according to either
`AIC <https://en.wikipedia.org/wiki/Akaike_information_criterion>`_,
`AICc <https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc>`_,
`BIC <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_ or
`HQIC <https://en.wikipedia.org/wiki/Hannan–Quinn_information_criterion>`_.
The function performs a search (either stepwise or parallelized)
over possible model orders within the constraints provided.
