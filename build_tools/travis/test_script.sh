#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

# License: 3-clause BSD

set -e

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import multiprocessing as mp; print('%d CPUs' % mp.cpu_count())"

run_tests() {
    TEST_CMD="pytest --showlocals --durations=20 --pyargs"

    # Get into a temp directory to run test from the installed pmdarima and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR

    # We need the setup.cfg for the nose settings
    cp setup.cfg $TEST_DIR
    cp .coveragerc $TEST_DIR
    cd $TEST_DIR

    if [[ "$COVERAGE" == "true" ]]; then
        TEST_CMD="$TEST_CMD --cov-config .coveragerc --cov pmdarima"
    fi
    $TEST_CMD pmdarima

    # Going back to git checkout folder needed for make test-doc
    cd $OLDPWD
}

if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi
