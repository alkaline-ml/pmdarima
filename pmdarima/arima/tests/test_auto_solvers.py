# -*- coding: utf-8 -*-

from pmdarima.arima import _auto_solvers as solvers
from pmdarima.compat.pytest import pytest_error_str

import numpy as np
import pytest


@pytest.mark.parametrize(
    'results_dict,ic_dict,expected', [

        # No nones, no overlap in IC
        pytest.param(
            {'foo_key': 'foo', 'bar_key': 'bar', 'baz_key': 'baz'},
            {'foo_key': 1.0, 'bar_key': 3.0, 'baz_key': 2.0},
            ['foo', 'baz', 'bar'],
        ),

        # we filter out Nones
        pytest.param(
            {'foo_key': 'foo', 'bar_key': 'bar', 'baz_key': None},
            {'foo_key': 1.0, 'bar_key': 3.0, 'baz_key': np.inf},
            ['foo', 'bar'],
        ),

    ]
)
def test_sort_and_filter_valid(results_dict, ic_dict, expected):
    actual = solvers._sort_and_filter_fits(results_dict, ic_dict)
    assert tuple(expected) == tuple(actual), \
        "\nExpected: %r" \
        "\nActual: %r" \
        % (expected, actual)


def test_sort_and_filter_error():
    results_dict = {'foo_key': None, 'bar_key': None, 'baz_key': None}
    ic_dict = {'foo_key': np.inf, 'bar_key': np.inf, 'baz_key': np.inf}

    with pytest.raises(ValueError) as ve:
        solvers._sort_and_filter_fits(results_dict, ic_dict)
    assert "no-successful-model" in pytest_error_str(ve)
