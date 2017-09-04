.. pyramid documentation master file, created by
   sphinx-quickstart on Sun Sep  3 15:16:29 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================================
Pyramid: ARIMA estimators for Python
====================================
Pyramid brings R's beloved ``auto.arima`` to Python, making an even stronger
case for why you don't need R for data science. It does so not by calling R
under the hood, but by wrapping statsmodels' well-tested
`ARIMA <http://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMA.html>`_ and
`SARIMAX <http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html>`_
estimators in a single, easy-to-use scikit-learn-esque estimator.

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2

   Setup <./setup.rst>
   Quickstart <./quickstart.rst>
   Codebase <./codebase.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
