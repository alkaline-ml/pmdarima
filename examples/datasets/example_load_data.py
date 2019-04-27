"""
===============
Dataset loading
===============


In this example, we demonstrate pyramid's built-in toy datasets that can be
used for benchmarking or experimentation. Pyramid has several built-in datasets
that exhibit seasonality, non-stationarity, and other time series nuances.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pmdarima as pm

# #############################################################################
# You can load the datasets via load_<name>
lynx = pm.datasets.load_lynx()
print("Lynx array:")
print(lynx)

# You can also get a series, if you rather
print("\nLynx series head:")
print(pm.datasets.load_lynx(as_series=True).head())

# Several other datasets:
air_passengers = pm.datasets.load_airpassengers()
austres = pm.datasets.load_austres()
heart_rate = pm.datasets.load_heartrate()
wineind = pm.datasets.load_wineind()
woolyrnq = pm.datasets.load_woolyrnq()
