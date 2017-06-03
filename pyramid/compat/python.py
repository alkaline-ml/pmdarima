# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Provide compatibility between differing versions of Python

from __future__ import absolute_import
import sys

# is this python 3?
PY3 = sys.version_info.major == 3

if PY3:
    xrange = range
    long = int

else:
    xrange = xrange
    long = long
