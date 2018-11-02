# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Provide compatibility between differing versions of Python

from __future__ import absolute_import

import sys
import os

# is this python 3?
PY3 = sys.version_info.major == 3

if PY3:
    xrange = range
    long = int

# Python 2.7
else:
    xrange = xrange
    long = long


def safe_mkdirs(loc):
    """Safely create a directory, even if it exists.

    Using ``os.makedirs`` can raise an OSError if a directory already exists.
    This safely attempts to create a directory, and passes if it already
    exists. It also safely avoids the race condition of checking for the
    directory's existence prior to creating it.

    Parameters
    ----------
    loc : str or unicode
        The absolute path to the directory to create.
    """
    try:
        os.makedirs(loc)
    # since this is a race condition, just try to make it
    except OSError as e:
        # Anything OTHER than the dir already exists error
        if e.errno != 17:
            raise
