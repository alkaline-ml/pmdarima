# -*- coding: utf-8 -*-

from pmdarima.datasets._base import load_date_example
from pmdarima.preprocessing.exog import DateFeaturizer
from pmdarima.compat.pytest import pytest_error_str

from numpy.testing import assert_array_equal
import pytest

y, X = load_date_example()


def test_no_options_warns():
    feat = DateFeaturizer(column_name="date",
                          with_day_of_month=False,
                          with_day_of_week=False)

    with pytest.warns(UserWarning) as w:
        y_prime, X_prime = feat.fit_transform(y, X)

    assert w is not None
    assert_array_equal(y, y_prime)
    assert X.equals(X_prime)


def test_illegal_column_fails():
    X_prime = X.copy()
    X_prime["date2"] = X_prime["date"].astype(str)

    feat = DateFeaturizer(column_name="date2")
    with pytest.raises(ValueError) as ve:
        feat.fit_transform(y, X_prime)

    assert "pd.Timestamp type" in pytest_error_str(ve)


def test_missing_column_fails():
    feat = DateFeaturizer(column_name="date2")
    with pytest.raises(ValueError) as ve:
        feat.fit_transform(y, X)

    assert "must exist" in pytest_error_str(ve)


def test_numpy_array_fails():
    feat = DateFeaturizer(column_name="date")
    with pytest.raises(TypeError) as te:
        feat.fit_transform(y, X.values)

    assert "X must be" in pytest_error_str(te)


def _dummy_assertions(X_prime):
    # they are dummies, so they should sum to 1 along the row axis
    dummies = X_prime[[n for n in X_prime.columns if 'WEEKDAY' in n]]
    assert (dummies.values.sum(axis=1) == 1).all()


def _ordinal_assertions(X_prime):
    # it's the day of the month, so they should all be > 0
    series = X_prime["DATE-DAY-OF-MONTH"]
    assert (series.values.ravel() > 0).all()


def test_all_true():
    feat = DateFeaturizer(column_name="date",
                          with_day_of_month=True,
                          with_day_of_week=True)

    y_prime, X_prime = feat.fit_transform(y, X)

    assert_array_equal(y, y_prime)
    assert y is not y_prime

    # there should be 8 columns in the X_prime (7 for days of the week, 1 for
    # ordinal)
    assert X_prime.shape[1] == 8

    _dummy_assertions(X_prime)
    _ordinal_assertions(X_prime)

    # date column should not be there anymore
    assert "date" not in X_prime.columns.tolist()


def test_dummy_only():
    feat = DateFeaturizer(column_name="date",
                          prefix="DATE",
                          with_day_of_month=False,
                          with_day_of_week=True)

    y_prime, X_prime = feat.fit_transform(y, X)

    assert_array_equal(y, y_prime)
    assert y is not y_prime

    # there should be 7 columns in the X_prime (7 for days of the week)
    assert X_prime.shape[1] == 7

    _dummy_assertions(X_prime)

    # show ordinal col not here
    assert "DATE-DAY-OF-MONTH" not in X_prime.columns.tolist()

    # date column should not be there anymore
    assert "date" not in X_prime.columns.tolist()


def test_ordinal_only():
    feat = DateFeaturizer(column_name="date",
                          prefix="DATE",
                          with_day_of_month=True,
                          with_day_of_week=False)

    y_prime, X_prime = feat.fit_transform(y, X)

    assert_array_equal(y, y_prime)
    assert y is not y_prime

    # there should be 1 column in the X_prime df
    assert X_prime.shape[1] == 1

    _ordinal_assertions(X_prime)

    # show ordinal col not here
    assert not [n for n in X_prime.columns.tolist() if "WEEKDAY" in n]

    # date column should not be there anymore
    assert "date" not in X_prime.columns.tolist()
