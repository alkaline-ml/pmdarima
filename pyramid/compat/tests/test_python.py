# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pyramid.compat.python import long, xrange, safe_mkdirs

import shutil
import os


def test_xrange_compatibility():
    for _ in xrange(5):
        pass


def test_long_compatibility():
    assert long(5) > 4  # show this works on both


def test_safe_mkdirs():
    folder = 'fake_dir'
    try:
        # Assert a folder isn't already there
        assert not os.path.exists(folder)

        # Create a folder
        safe_mkdirs(folder)
        assert os.path.exists(folder)

        # Try to create it again, show we don't break down
        safe_mkdirs(folder)
    finally:
        # Always remove it
        if os.path.exists(folder):
            shutil.rmtree(folder)
