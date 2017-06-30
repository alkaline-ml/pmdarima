
from __future__ import absolute_import, print_function
from pyramid.utils.metaestimators import if_has_delegate


class _IfHasDelegateTester(object):
    def __init__(self):
        pass

    def fit(self):
        self.a_ = None
        return self

    @if_has_delegate('a_')
    def predict(self):
        return True

    @if_has_delegate(['b_', 'a_'])
    def predict2(self):
        return True


def test_single_delegate():
    # show it passes for a "fit"
    assert _IfHasDelegateTester().fit().predict()
    assert not hasattr(_IfHasDelegateTester(), 'predict')


def test_multiple_delegates():
    assert _IfHasDelegateTester().fit().predict2()
    assert not hasattr(_IfHasDelegateTester(), 'predict2')
