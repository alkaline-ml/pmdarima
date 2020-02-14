"""
=========================
Persisting an ARIMA model
=========================


This example demonstrates how we can persist an ARIMA model to disk after
fitting it. It can then be loaded back up and used to generate forecasts.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pmdarima as pm
from pmdarima import model_selection
import joblib  # for persistence
import os

# #############################################################################
# Load the data and split it into separate pieces
y = pm.datasets.load_wineind()
train, test = model_selection.train_test_split(y, train_size=125)

# Fit an ARIMA
arima = pm.ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
arima.fit(y)

# #############################################################################
# Persist a model and create predictions after re-loading it
pickle_tgt = "arima.pkl"
try:
    # Pickle it
    joblib.dump(arima, pickle_tgt, compress=3)

    # Load the model up, create predictions
    arima_loaded = joblib.load(pickle_tgt)
    preds = arima_loaded.predict(n_periods=test.shape[0])
    print("Predictions: %r" % preds)

finally:
    # Remove the pickle file at the end of this example
    try:
        os.unlink(pickle_tgt)
    except OSError:
        pass
