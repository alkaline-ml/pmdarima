.. _setup:

=====
Setup
=====

Pmdarima depends on several prominent python packages:

* `Numpy <https://github.com/numpy/numpy>`_ (>=1.17.3)
* `SciPy <https://github.com/scipy/scipy>`_ (>=1.3.2)
* `Scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ (>=0.22)
* `Pandas <https://github.com/pandas-dev/pandas>`_ (>=0.19)
* `Statsmodels <https://github.com/statsmodels/statsmodels>`_ (>=0.11)

Install from PyPi
-----------------

Pmdarima is on pypi under the package name ``pmdarima`` and can be
downloaded via ``pip``:

.. code-block:: bash

    $ pip install pmdarima

Pmdarima uses Cython, which means there is some C source that was built in
the distribution process. To ensure the package was built correctly, import
the following module in python:

.. code-block:: python

    from pmdarima.arima import auto_arima

If you encounter an ``ImportError``, try updating numpy and re-installing. Outdated
numpy versions have been observed to break the pmdarima build.

Install from Conda
------------------

Pmdarima is on conda under the channel ``conda-forge`` and package name ``pmdarima``. It can be downloaded via ``conda``
like so:

.. code-block:: bash

    $ conda config --add channels conda-forge
    $ conda config --set channel_priority strict
    $ conda install pmdarima

**Note:** We do not maintain our own Conda binaries, they are maintained at https://github.com/conda-forge/pmdarima-feedstock.
See that repo for further documentation on working with Pmdarima on Conda.

Build from source
-----------------

If you'd like to install a development or bleeding edge version of pmdarima,
you can always build it from the git source. First clone it from Git:

.. code-block:: bash

    $ git clone https://github.com/alkaline-ml/pmdarima.git
    $ cd pmdarima

Building the package will require ``gcc`` (unix) or a Windows equivalent, like
``MinGW``. To build in development mode (for running unit tests):

.. code-block:: bash

    $ python setup.py develop

You can also use the ``Makefile`` if you're on a posix machine:

.. code-block:: bash

    $ make develop

Alternatively, to install the package in your ``site-packages``:

.. code-block:: bash

    $ python setup.py install

Or, with the ``Makefile``:

.. code-block:: bash

    $ make install
