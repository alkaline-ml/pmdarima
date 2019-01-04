# -*- coding: utf-8 -*-
#
# Author: Taylor Smith <taylor.smith@alkaline-ml.com>
#
# Provide compatibility between differing versions of Python

from __future__ import absolute_import

import os

# We only ever support python 3 now, so no 'if' required anymore.
xrange = range
long = int


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
    # try:
    #     os.makedirs(loc)
    # # since this is a race condition, just try to make it
    # except OSError as e:
    #     # Anything OTHER than the dir already exists error
    #     if e.errno != 17:
    #         raise
    return os.makedirs(loc, exist_ok=True)
