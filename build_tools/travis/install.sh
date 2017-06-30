#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Install miniconda
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    MINICONDA_PATH=/home/travis/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
    export PATH=$MINICONDA_PATH/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    if [[ "$PYTHON_VERSION" == "2.7" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION \
            numpy scipy cython=$CYTHON_VERSION statsmodels \
            scikit-learn=$SCIKIT_LEARN_VERSION

    elif [[ "$INSTALL_MKL" == "true" ]]; then
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose pytest \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
            mkl cython=$CYTHON_VERSION \
            scikit-learn=$SCIKIT_LEARN_VERSION \
            statsmodels=$STATSMODELS_VERSION

    else
        conda create -n testenv --yes python=$PYTHON_VERSION pip nose pytest \
            numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
            nomkl cython=$CYTHON_VERSION \
            scikit-learn=$SCIKIT_LEARN_VERSION \
            statsmodels=$STATSMODELS_VERSION
    fi
    source activate testenv

    # Install nose-timer via pip
    pip install nose-timer
fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage codecov coveralls
fi

if [[ "$SKIP_TESTS" == "true" ]]; then
    echo "No need to build pyramid when not running the tests"
else
    # Build pyramid in the install.sh script to collapse the verbose
    # build output in the travis output when it succeeds.
    python --version
    python -c "import numpy; print('numpy %s' % numpy.__version__)"
    python -c "import scipy; print('scipy %s' % scipy.__version__)"
    python -c "\
try:
    import pandas
    print('pandas %s' % pandas.__version__)
except ImportError:
    pass
"
    python setup.py develop
    ccache --show-stats
fi
