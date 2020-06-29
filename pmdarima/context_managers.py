# -*- coding: utf-8 -*-

import contextlib

__all__ = ['except_and_reraise']


@contextlib.contextmanager
def except_and_reraise(*except_errs, raise_err=None, raise_msg=None):
    """Catch a lower-level error and re-raise with a more meaningful message

    In some cases, Numpy linalg errors can be raised in perplexing spots. This
    allows us to catch the lower-level errors in spots where we are aware of
    them so that we may raise with a more meaningful message.

    Parameters
    ----------
    *except_errs : var-args, BaseException
        A variable list of exceptions to catch

    raise_err : BaseException, Error
        The exception to raise

    raise_msg : str
        The message to raise
    """
    if raise_err is None:
        raise TypeError("raise_err must be used as a key-word arg")
    if raise_msg is None:
        raise TypeError("raise_msg must be used as a key-word arg")

    try:
        yield
    except except_errs as e:
        message = "%s (raised from %s: %s)" \
                  % (raise_msg,
                     e.__class__.__name__,
                     str(e))
        raise raise_err(message)
