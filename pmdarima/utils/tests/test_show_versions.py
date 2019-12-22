# -*- coding: utf-8 -*-

import pmdarima as pm
from pmdarima.utils._show_versions import _get_deps_info


# Just show it doesn't blow up...
def test_show_versions():
    pm.show_versions()


def test_show_versions_when_not_present():
    deps = ['big-ol-fake-pkg']
    assert _get_deps_info(deps=deps)['big-ol-fake-pkg'] is None
