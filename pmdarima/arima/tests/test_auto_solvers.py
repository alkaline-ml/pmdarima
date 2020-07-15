# -*- coding: utf-8 -*-

from pmdarima.arima import _auto_solvers as solvers
from pmdarima.compat.pytest import pytest_error_str

import numpy as np
import pytest


@pytest.mark.parametrize(
    'models,expected', [

        # No nones, no overlap in IC
        pytest.param(
            [('foo', 'time', 1.0),
             ('bar', 'time', 3.0),
             ('baz', 'time', 2.0)],
            ['foo', 'baz', 'bar'],
        ),

        # we filter out Nones and infs
        pytest.param(
            [('foo', 'time', 1.0),
             ('bar', 'time', 3.0),
             ('baz', 'time', np.inf),
             (None, 'time', 0.0)],
            ['foo', 'bar'],
        ),

    ]
)
def test_sort_and_filter_fits_valid(models, expected):
    actual = solvers._sort_and_filter_fits(models)
    assert tuple(expected) == tuple(actual), \
        "\nExpected: %r" \
        "\nActual: %r" \
        % (expected, actual)


def test_sort_and_filter_fits_error():
    results = [(None, 'time', 1.0), ('foo', 'time', np.inf)]

    with pytest.raises(ValueError) as ve:
        solvers._sort_and_filter_fits(results)
    assert "no-successful-model" in pytest_error_str(ve)
