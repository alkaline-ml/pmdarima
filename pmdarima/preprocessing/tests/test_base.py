# -*- coding: utf-8 -*-

import pytest
from pmdarima.preprocessing import base
from pmdarima.compat.pytest import pytest_error_str


def test_value_error_on_update_check():
    with pytest.raises(ValueError) as ve:
        base.UpdatableMixin()._check_endog(None)
    assert 'cannot be None' in pytest_error_str(ve)
