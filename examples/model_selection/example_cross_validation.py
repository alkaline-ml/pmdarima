"""
========================================
Cross-validating your time series models
========================================


Like scikit-learn, ``pmdarima`` provides several different strategies for
cross-validating your time series models. The interface was designed to behave
as similarly as possible to that of scikit to make its usage as simple as
possible.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import numpy as np
import pmdarima as pm
from pmdarima import model_selection
from pmdarima import utils

print("pmdarima version: %s" % pm.__version__)

# Load the data and split it into separate pieces
data = pm.datasets.load_wineind()
train, test = data[:165], data[165:]

# Even though we have a dedicated train/test split, we can (and should) still
# use cross-validation on our training set to get a good estimate of the model
# performance. We can choose which model is better based on how it performs
# over various folds.
model1 = pm.ARIMA(order=(2, 1, 1), seasonal_order=(0, 0, 0, 1))
model2 = pm.ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
cv = model_selection.SlidingWindowForecastCV(window_size=100, step=24, h=1)

# store the residuals for each model
m1_residuals = []
m2_residuals = []

for train_window_indices, val_index in cv.split(train):
    tr_fold = train[train_window_indices]
    model1.fit(tr_fold)
    model2.fit(tr_fold)

    m1_residuals.append(train[val_index] - model1.predict(n_periods=1))
    m2_residuals.append(train[val_index] - model2.predict(n_periods=1))

# make sure residuals are flat and compute RMSE
rmse = lambda arr: np.sqrt(np.average(utils.check_endog(arr) ** 2))
errors = [rmse(m1_residuals), rmse(m2_residuals)]
models = [model1, model2]

# print out the answer
better_index = np.argmin(errors)  # type: int
print("Lowest RMSE: {}".format(errors[better_index]))
print("Best model: {}".format(models[better_index]))
