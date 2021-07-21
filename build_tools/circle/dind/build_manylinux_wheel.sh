#!/bin/bash

# EXECUTED IN QUAY DOCKER IMAGE WHERE /io IS A MOUNTED VOLUME OF PMDARIMA ROOT

# Modify permissions on file
set -e -x

# Compile wheels
PYTHON="/opt/python/${PYTHON_VERSION}/bin/python"
PIP="/opt/python/${PYTHON_VERSION}/bin/pip"

# We have to use wheel < 0.32 since they inexplicably removed the open_for_csv
# function from the package after 0.31.1 and it fails for Python 3.6?!
${PIP} install --upgrade pip wheel==0.31.1
${PIP} install --upgrade "setuptools>=38.6.0,!=50.0.0"

# NOW we can install requirements
${PIP} install -r /io/requirements.txt
make -C /io/ PYTHON="${PYTHON}"

# Make sure the VERSION file is present for this. For whatever reason, the
# make -C call removes it
echo ${PMDARIMA_VERSION} > /io/pmdarima/VERSION
${PIP} wheel /io/ -w /io/dist/

# Bundle external shared libraries into the wheels.
for whl in /io/dist/*.whl; do
    if [[ "$whl" =~ "pmdarima" ]]; then
        auditwheel repair $whl -w /io/dist/ #repair pmdarima wheel and output to /io/dist
    fi

    rm $whl # remove wheel
done
