#!/bin/bash

# Dependencies
pip install numpy
pip install scipy scikit-learn pandas statsmodels

# Any other requirements
pip install -r requirements.txt

# Install the old version of the package
pip install pyramid-arima

# Setup the new package
python setup.py install
mkdir testing_dir && cd testing_dir

# Fit an arima and pickle it somewhere
python -c "\
from sklearn.externals import joblib
import pyramid as pm

lynx = pm.datasets.load_lynx()
arima = pm.auto_arima(lynx)
joblib.dump(arima, 'model.pkl')
"

# Show we can import it with the new namespace
python -c "\
print('Test loading old model with pmdarima:')
import pmdarima as pm
from sklearn.externals import joblib
modl = joblib.load('model.pkl')
print(modl.predict(n_periods=5))
"

status=$?
exit $status
