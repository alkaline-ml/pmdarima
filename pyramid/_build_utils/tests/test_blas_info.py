# this is really just here for the coverage. The only
# time this function is ever called is when we're running setup

from pyramid._build_utils import get_blas_info


def test_blas():
    _ = get_blas_info()
