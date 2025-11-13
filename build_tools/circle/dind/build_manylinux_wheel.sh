#!/bin/bash

# EXECUTED IN QUAY DOCKER IMAGE WHERE /io IS A MOUNTED VOLUME OF PMDARIMA ROOT

# Modify permissions on file
set -e -x

# Compile wheels
PYTHON="/opt/python/${PYTHON_VERSION}/bin/python"
PIP="/opt/python/${PYTHON_VERSION}/bin/pip"

${PIP} install --upgrade pip build
cd /io
${PYTHON} -m build --wheel

# Bundle external shared libraries into the wheels.
for whl in /io/dist/*.whl; do
    if [[ "$whl" =~ "pmdarima" ]]; then
        auditwheel repair $whl -w /io/dist/ #repair pmdarima wheel and output to /io/dist
    fi

    rm $whl # remove wheel
done
