# -*- coding: utf-8 -*-


def pytest_error_str(error):
    """Different for different versions of Pytest"""
    try:
        return str(error.value)
    except AttributeError:
        return str(error)


def pytest_warning_messages(warnings):
    """Get the warning messages for captured warnings"""
    return [str(w.message) for w in warnings.list]
