# -*- coding: utf-8 -*-

import contextlib

__all__ = ['except_and_reraise']


@contextlib.contextmanager
def except_and_reraise(except_err, except_msg, raise_err, raise_msg):
    """Catch a lower-level error and re-raise with a more meaningful message

    In some cases, Numpy linalg errors can be raised in perplexing spots. This
    allows us to catch the lower-level errors in spots where we are aware of
    them so that we may raise with a more meaningful message.
    """
    try:
        yield
    except except_err as e:
        if except_msg in str(e):
            message = "%s (raised from %s: %s)" \
                      % (raise_msg,
                         except_err.__name__,
                         str(e))
            raise raise_err(message)

        # otherwise raise it raw
        raise
