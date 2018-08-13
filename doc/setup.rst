.. _setup:

=====
Setup
=====

Pyramid depends on several prominent python packages:

* `Numpy <https://github.com/numpy/numpy>`_
* `SciPy <https://github.com/scipy/scipy>`_
* `Scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ (>= 0.17)
* `Pandas <https://github.com/pandas-dev/pandas>`_
* `Statsmodels <https://github.com/statsmodels/statsmodels>`_ (>=0.9.0)

Installation
------------

Pyramid is on pypi under the package name ``pyramid-arima`` and can be
downloaded via ``pip``:

.. code-block:: bash

    $ pip install pyramid-arima

To ensure the package was built correctly, import the following module in
python:

.. code-block:: python

    from pyramid.arima import auto_arima

If you encounter an ``ImportError``, try updating numpy and re-installing. Outdated
numpy versions have been observed to break the Pyramid build.
