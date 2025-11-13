"""
====================================
Displaying key timeseries statistics
====================================


Visualizing characteristics of a time series is a key component to effective
forecasting. In this example, we'll look at a very simple method to examine
critical statistics of a time series object.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pmdarima as pm
from pmdarima import datasets
from pmdarima import preprocessing

# We'll use the sunspots dataset for this example
y = datasets.load_sunspots(True)
print("Data shape: {}".format(y.shape[0]))
print("Data head:")
print(y.head())

# Let's look at the series, its ACF plot, and a histogram of its values
pm.tsdisplay(y, lag_max=90, title="Sunspots", show=True)

# Notice that the histogram is very skewed. This is a prime candidate for
# box-cox transformation
y_bc, _ = preprocessing.BoxCoxEndogTransformer(lmbda2=1e-6).fit_transform(y)
pm.tsdisplay(
    y_bc, lag_max=90, title="Sunspots (BoxCox-transformed)", show=True)

print("""
As evidenced by the more normally distributed values in the transformed series,
using a Box-Cox transformation may prove useful prior to fitting your model.
""")
