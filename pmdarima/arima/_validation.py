# -*- coding: utf-8 -*-


def check_trace(trace):
    """Check the value of trace"""
    if trace is None:
        return 0
    if isinstance(trace, (int, bool)):
        return int(trace)
    # otherwise just be truthy with it
    if trace:
        return 1
    return 0
