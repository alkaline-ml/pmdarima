#!/usr/bin/env bash
set -x
set -e

# Don't test with Conda here, use virtualenv instead
pip install virtualenv

if command -v pypy3; then
    virtualenv -p $(command -v pypy3) pypy-env
elif command -v pypy; then
    virtualenv -p $(command -v pypy) pypy-env
fi

source pypy-env/bin/activate

python --version
which python

pip install --extra-index https://antocuni.github.io/pypy-wheels/ubuntu numpy Cython pytest
pip install scipy
pip install scikit-learn
# Pandas has starting throwing issues in Pypy now...
pip install "pandas==0.23.*" statsmodels matplotlib
pip install --extra-index https://antocuni.github.io/pypy-wheels/ubuntu pytest-mpl pytest-benchmark

ccache -M 512M
export CCACHE_COMPRESS=1
export PATH=/usr/lib/ccache:$PATH
export LOKY_MAX_CPU_COUNT="2"

pip install -vv -e .

# Pytest is known to consume lots of memory for a large number of tests,
# and Circle 2.0 limits 4GB per container.
python -m pytest pmdarima/ -p no:logging --mpl --mpl-baseline-path=etc/pytest_images --benchmark-skip
