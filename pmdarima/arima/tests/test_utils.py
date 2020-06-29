# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pmdarima.arima.utils import nsdiffs
from pmdarima.compat.pytest import pytest_warning_messages


def test_issue_351():
    y = np.array([
        1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 1, 6, 2, 1, 0,
        2, 0, 1, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 6,
        0, 0, 0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0
    ])

    with pytest.warns(UserWarning) as w_list:
        D = nsdiffs(y, m=52, max_D=2, test='ocsb')

    assert D == 1

    warnings_messages = pytest_warning_messages(w_list)
    assert len(warnings_messages) == 1
    assert 'shorter than m' in warnings_messages[0]
