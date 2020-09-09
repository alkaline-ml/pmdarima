# -*- coding: utf-8 -*-

import contextlib
import pytest


def pytest_error_str(error):
    """Different for different versions of Pytest"""
    try:
        return str(error.value)
    except AttributeError:
        return str(error)


def pytest_warning_messages(warnings):
    """Get the warning messages for captured warnings"""
    return [str(w.message) for w in warnings.list]


@contextlib.contextmanager
def raises(exception):
    """Allows context managers for catching NO errors"""
    if exception is None:
        yield None

    else:
        with pytest.raises(exception) as e:
            yield e
