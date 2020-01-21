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
    import joblib
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

