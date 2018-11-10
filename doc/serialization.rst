.. _serializing:

=============================
Serializing your ARIMA models
=============================

After you've fit your model and you're ready to start making predictions out
in your production environment, it's time to save your ARIMA to disk.
Pmdarima models can be serialized with ``pickle`` or ``joblib``, just as with
most other python objects:

.. code-block:: python

    from pmdarima.arima import auto_arima
    from pmdarima.datasets import load_lynx
    import numpy as np

    # For serialization:
    from sklearn.externals import joblib
    import pickle

    # Load data and fit a model
    y = load_lynx()
    arima = auto_arima(y, seasonal=True)

    # Serialize with Pickle
    with open('arima.pkl', 'wb') as pkl:
        pickle.dump(arima, pkl)

    # You can still make predictions from the model at this point
    arima.predict(n_periods=5)

    # Now read it back and make a prediction
    with open('arima.pkl', 'rb') as pkl:
        pickle_preds = pickle.load(pkl).predict(n_periods=5)

    # Or maybe joblib tickles your fancy
    joblib.dump(arima, 'arima.pkl')
    joblib_preds = joblib.load('arima.pkl').predict(n_periods=5)

    # show they're the same
    np.allclose(pickle_preds, joblib_preds)

If your job is to build models, that's probably all you really care to know about
the serialization process. However, there are several intricacies of how pmdarima
internally saves a model that you might care to know for development purposes.


Intricacies of ARIMA serialization
----------------------------------

The ARIMA class is a generalization of three models:

  * ``statsmodels.tsa.ARMA``
  * ``statsmodels.tsa.ARIMA``
  * ``statsmodels.tsa.statespace.SARIMAX``

The ``statsmodels`` library does not play very nicely with pickling, so under
the hood the pmdarima ARIMA class does some monkey-patching.


The serialization process
~~~~~~~~~~~~~~~~~~~~~~~~~

When the pickling process begins, the ARIMA class will first save the internal
model into a directory defined by the ``pmdarima._config.PYRAMID_ARIMA_CACHE``
variable (default is ``.pyramid-arima-cache/``). Next, it will pickle the class
instance to the defined location, save the location as a temporary attribute,
and re-attach the model state to the instance so that you can continue to make
predictions or otherwise use the ARIMA model after pickling.

The de-serialization process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When unpickling an ARIMA, the class instance is unpickled first, and then the
internal ``statsmodels`` object is loaded from the cached directory, re-attached
to the model state, and returned.
