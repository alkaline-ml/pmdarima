# -*- coding: utf-8 -*-

from pmdarima.arima.auto import StepwiseContext, auto_arima
from pmdarima.arima._context import ContextStore, ContextType
from pmdarima.arima import _context as context_lib
from pmdarima.datasets import load_lynx, load_wineind
from unittest import mock
import threading
import collections
import pytest

lynx = load_lynx()
wineind = load_wineind()


# test StepwiseContext parameter validation
@pytest.mark.parametrize(
    'max_steps,max_dur', [
        pytest.param(-1, None),
        pytest.param(0, None),
        pytest.param(1001, None),
        pytest.param(1100, None),
        pytest.param(None, -1),
        pytest.param(None, 0),
    ])
def test_stepwise_context_args(max_steps, max_dur):
    with pytest.raises(ValueError):
        StepwiseContext(max_steps=max_steps, max_dur=max_dur)


# test auto_arima stepwise run with  StepwiseContext
def test_auto_arima_with_stepwise_context():
    samp = lynx[:8]
    with StepwiseContext(max_steps=3, max_dur=30):
        with pytest.warns(UserWarning) as uw:
            auto_arima(samp, suppress_warnings=False, stepwise=True,
                       error_action='ignore')

            # assert that max_steps were taken
            assert any(str(w.message)
                       .startswith('stepwise search has reached the '
                                   'maximum number of tries') for w in uw)


# test effective context info in nested context scenario
def test_nested_context():
    ctx1_data = {'max_dur': 30}
    ctx2_data = {'max_steps': 5}
    ctx1 = StepwiseContext(**ctx1_data)
    ctx2 = StepwiseContext(**ctx2_data)

    with ctx1, ctx2:
        effective_ctx_data = ContextStore.get_or_empty(
            ContextType.STEPWISE)
        expected_ctx_data = ctx1_data.copy()
        expected_ctx_data.update(ctx2_data)

        assert all(effective_ctx_data[key] == expected_ctx_data[key]
                   for key in expected_ctx_data.keys())

        assert all(effective_ctx_data[key] == expected_ctx_data[key]
                   for key in effective_ctx_data.keys())


# Test a context honors the max duration
def test_max_dur():
    # set arbitrarily low to guarantee will always pass after one iter
    with StepwiseContext(max_dur=.5), \
            pytest.warns(UserWarning) as uw:

        auto_arima(lynx, stepwise=True)
        # assert that max_dur was reached
        assert any(str(w.message)
                   .startswith('early termination') for w in uw)


# Test that a context after the first will not inherit the first's attrs
def test_subsequent_contexts():
    # Force a very fast fit
    with StepwiseContext(max_dur=.5), \
            pytest.warns(UserWarning):
        auto_arima(lynx, stepwise=True)

    # Out of scope, should be EMPTY
    assert ContextStore.get_or_empty(ContextType.STEPWISE).get_type() \
        is ContextType.EMPTY

    # Now show that we DON'T hit early termination by time here
    with StepwiseContext(max_steps=100), \
            pytest.warns(UserWarning) as uw:

        ctx = ContextStore.get_or_empty(ContextType.STEPWISE)
        assert ctx.get_type() is ContextType.STEPWISE
        assert ctx.max_dur is None

        auto_arima(lynx, stepwise=True)
        # assert that max_dur was NOT reached
        assert not any(str(w.message)
                       .startswith('early termination') for w in uw)


# test param validation of ContextStore's add, get and remove members
def test_add_get_remove_context_args():
    with pytest.raises(ValueError):
        ContextStore._add_context(None)

    with pytest.raises(ValueError):
        ContextStore._remove_context(None)

    with pytest.raises(ValueError):
        ContextStore.get_context(None)


def test_context_store_accessible_across_threads():
    # Make sure it's completely empty by patching it
    d = {}
    with mock.patch('pmdarima.arima._context._ctx.store', d):

        # pushes onto the Context Store
        def push(n):
            # n is the number of times this has been executed before. If > 0,
            # assert there is a context there
            if n > 0:
                assert len(context_lib._ctx.store[ContextType.STEPWISE]) == n
            else:
                context_lib._ctx.store[ContextType.STEPWISE] = \
                    collections.deque()

            new_ctx = StepwiseContext()
            context_lib._ctx.store[ContextType.STEPWISE].append(new_ctx)
            assert len(context_lib._ctx.store[ContextType.STEPWISE]) == n + 1

        for i in range(5):
            t = threading.Thread(target=push, args=(i,))
            t.start()
            t.join(1)  # it shouldn't take even close to this time

    # Assert the mock has lifted
    assert context_lib._ctx.store is not d
