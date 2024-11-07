"""
==================
Array differencing
==================


In this example, we demonstrate pyramid's array differencing, and how it's used
in conjunction with the ``d`` term to lag a time series.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

from pmdarima.utils import array

# Build an array and show first order differencing results
x = array.c(10, 4, 2, 9, 34)
lag_1 = array.diff(x, lag=1, differences=1)

# The result will be the same as: x[1:] - x[:-1]
print(lag_1)  # [-6., -2., 7., 25.]

# Note that lag and differences are not the same! If we crank diff up by one,
# it performs the same differencing as above TWICE. Lag, therefore, controls
# the number of steps backward the ts looks when it differences, and the
# `differences` parameter controls how many times to repeat.
print(array.diff(x, lag=1, differences=2))  # [4., 9., 18.]

# Conversely, when we set lag to 2, the array looks two steps back for its
# differencing operation (only one).
print(array.diff(x, lag=2, differences=1))  # [-8., 5., 32.]

# The lag parameter is controlled by `m`, which is the seasonal periodicity of
# a time series. If your series is non-seasonal, lag will typically be 1.
