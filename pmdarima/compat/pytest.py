# -*- coding: utf-8 -*-


def pytest_error_str(error):
    """Different for different versions of Pytest"""
    try:
        return str(error.value)
    except AttributeError:
        return str(error)
