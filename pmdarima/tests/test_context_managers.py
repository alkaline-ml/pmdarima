# -*- coding: utf-8 -*-

from pmdarima import context_managers as ctx


def test_except_and_reraise_do_reraise():
    try:
        with ctx.except_and_reraise(
                ValueError, "foo message", KeyError, "bar message"
        ):
            raise ValueError("contains foo message")
    except KeyError as k:
        assert "bar message" in str(k)
        assert "raised from ValueError" in str(k)
    else:
        assert False, "Test failed"


def test_except_and_reraise_no_reraise():
    try:
        with ctx.except_and_reraise(
                ValueError, "foo message", KeyError, "bar message"
        ):
            raise ValueError("some other error")
    except ValueError as v:
        assert "some other error" in str(v)
    else:
        assert False, "Test failed"
