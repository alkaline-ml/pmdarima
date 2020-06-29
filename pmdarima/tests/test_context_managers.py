# -*- coding: utf-8 -*-

from pmdarima import context_managers as ctx
from pmdarima.compat.pytest import pytest_error_str

import pytest


def test_except_and_reraise_do_reraise():
    with pytest.raises(KeyError) as ke:
        with ctx.except_and_reraise(
                ValueError,
                raise_err=KeyError,
                raise_msg="bar message"
        ):
            raise ValueError("contains foo message")

    msg = pytest_error_str(ke)
    assert "bar message" in msg
    assert "raised from ValueError" in msg


def test_except_and_reraise_no_reraise():
    with pytest.raises(KeyError) as ke:
        with ctx.except_and_reraise(
                ValueError,
                raise_err=TypeError,
                raise_msg="bar message"
        ):
            raise KeyError("foo message")

    assert "foo message" in pytest_error_str(ke)


@pytest.mark.parametrize('err', [ValueError, KeyError, TypeError])
def test_multiple(err):

    class FooError(BaseException):
        pass

    with pytest.raises(FooError) as fe:
        with ctx.except_and_reraise(
                ValueError, KeyError, TypeError,
                raise_err=FooError,
                raise_msg="gotcha, fam",
        ):
            raise err("Boo!")

    assert "gotcha, fam" in pytest_error_str(fe)
