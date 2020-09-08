# -*- coding: utf-8 -*-

import pytest
from pmdarima.compat.pytest import pytest_error_str
from pmdarima.preprocessing.endog import LogEndogTransformer


def test_value_error_on_check():
    trans = LogEndogTransformer()  # could be anything, just need an instance
    with pytest.raises(ValueError) as ve:
        trans._check_y_X(None, None)
    assert 'non-None' in pytest_error_str(ve)
