#!/bin/bash
# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

# 02/10/2018 remove due to Travis build issue 6307
# set -e

# push coverage if necessary
if [[ "$COVERAGE" == "true" ]]; then
    echo "COVERAGE=true; moving .coverage to build dir"

    # Need to run coveralls from a git checkout, so we copy .coverage
    # from TEST_DIR where nosetests has been run
    rsync $TEST_DIR/.coverage $TRAVIS_BUILD_DIR --ignore-existing
    cd $TRAVIS_BUILD_DIR

    # Ignore coveralls failures as the coveralls server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
    echo "Running coverage"
    coveralls || echo "Coveralls upload failed"
fi

# make sure we have twine in case we deploy
echo "Installing twine"
pip install twine || "pip installing twine failed"

# remove the .egg-info dir so Mac won't bomb on bdist_wheel cmd (absolute path in SOURCES.txt)
echo "Removing egg info (if exists)"
rm -r pyramid_arima.egg-info/ || echo "No local .egg cache to remove"

# make a dist folder if not there, then make sure permissions are sufficient
mkdir -p dist
chmod 777 dist
