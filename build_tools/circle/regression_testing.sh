#!/bin/bash

# Dependencies
pip install numpy
pip install scipy scikit-learn pandas statsmodels

# Any other requirements
echo "installing requirements"
pip install -r requirements.txt

# Install the old version of the package
echo "installing old, deprecated version of the package"
pip install pyramid-arima

# Setup the new package
echo "setting up new package"
python setup.py install
mkdir testing_dir && cd testing_dir

# Fit an arima and pickle it somewhere
python -c "\
import joblib
import pyramid as pm

print('fitting and serializing model with old package')
wineind = pm.datasets.load_wineind()
arima = pm.auto_arima(wineind, seasonal=True, m=4, D=1)
joblib.dump(arima, 'model.pkl')
"

# Show we can import it with the new namespace
python -c "\
print('Test loading old model with pmdarima:')
import pmdarima as pm
import joblib
modl = joblib.load('model.pkl')
print(modl.predict(n_periods=5))
"

status=$?
exit $status
