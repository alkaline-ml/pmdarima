# -*- coding: utf-8 -*-
#
# re-entrant, reusable context manager to store execution context
#

import threading

# thread local value to store the context info
_ctx = threading.local()

class _context():
    """A generic context manager to store execution context.

    A generic, re-entrant, reusable context manager to store
    execution context in a threading.local instance. Has helper
    methods to iterate over the context info and provide a
    string representation of the context info.

    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(_ctx, key, kwargs[key])

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __getattr__(self, item):
        return getattr(_ctx, item, None)

    def __setattr__(self, key, value):
        setattr(_ctx, key, value)

    def __contains__(self, item):
        return hasattr(_ctx, item)

    def __getitem__(self, item):
        return getattr(_ctx, item, None)

    def __iter__(self):
        for key in [k for k in _ctx.__dir__() if not k.startswith('__')]:
            yield key

    def __repr__(self):
        return ', '.join(
            [key + ': ' + self[key]
                for key in _ctx.__dir__() if not key.startswith('__')
             ])
