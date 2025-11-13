"""
===================
Array concatenation
===================


In this example, we demonstrate pyramid's convenient ``c`` function, which is,
in essence, the same as R's. It's nothing more than a convenience function in
the package, but one you should understand if you're contributing.

.. raw:: html

   <br/>
"""
print(__doc__)

# Author: Taylor Smith <taylor.smith@alkaline-ml.com>

import pmdarima as pm
import numpy as np

# #############################################################################
# You can use the 'c' function to define an array from *args
array1 = pm.c(1, 2, 3, 4, 5)

# Or you can define an array from an existing iterable:
array2 = pm.c([1, 2, 3, 4, 5])
assert np.array_equal(array1, array2)

# You can even use 'c' to flatten arrays:
array_flat = pm.c(1, 2, 3, [4, 5])
assert np.array_equal(array_flat, np.arange(5) + 1)
