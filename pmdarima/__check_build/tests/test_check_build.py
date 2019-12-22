# -*- coding: utf-8 -*-

from pmdarima.__check_build import raise_build_error

import pytest


def test_raise_build_error():
    try:
        # Raise a value error to pass into the raise_build_error
        # to assert it turns it into an ImportError
        raise ValueError("this is a dummy err msg")
    except ValueError as v:
        with pytest.raises(ImportError):
            raise_build_error(v)
