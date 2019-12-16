# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Provide numpy compatibility and common variables. Since this
# is a relatively sparse script, I feel I must defend this design
# choice. See the docstring in the __init__: "Each sub-module is specifically
# designed not to make calls out to other portions of pmdarima and to
# remove circular dependencies."
#
# Since DTYPE is used commonly, this removes circular dependencies or
# hard-coding.

import numpy as np

# this is going to be the data-type used across pmdarima
DTYPE = np.float64
