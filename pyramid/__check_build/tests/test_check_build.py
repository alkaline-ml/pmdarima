# -*- coding: utf-8 -*-

from __future__ import absolute_import

from pyramid.__check_build import raise_build_error

from nose.tools import assert_raises


def test_raise_build_error():
    try:
        raise ValueError("this is a dummy err msg")
    except ValueError as v:
        assert_raises(ImportError, raise_build_error, v)
