"""
=======================
Simple auto_arima model
=======================


This is a simple example of how we can fit an ARIMA model in several lines
without knowing anything about our data or optimal hyper parameters.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pmdarima as pm
import numpy as np
from matplotlib import pyplot as plt

# #############################################################################
# Load the data and split it into separate pieces
data = pm.datasets.load_wineind()
train, test = data[:150], data[150:]

# Fit a simple auto_arima model
arima = pm.auto_arima(train, error_action='ignore', trace=1,
                      suppress_warnings=True,
                      seasonal=True, m=12)

# #############################################################################
# Plot actual test vs. forecasts:
x = np.arange(test.shape[0])
plt.scatter(x, test, marker='x')
plt.plot(x, arima.predict(n_periods=test.shape[0]))
plt.title('Actual test samples vs. forecasts')
plt.show()
