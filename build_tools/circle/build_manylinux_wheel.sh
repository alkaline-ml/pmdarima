#!/bin/bash

# Modify permissions on file
set -e -x

# Compile wheels
PYTHON="/opt/python/${PYTHON_VERSION}/bin/python"
PIP="/opt/python/${PYTHON_VERSION}/bin/pip"

# We have to use wheel < 0.32 since they inexplicably removed the open_for_csv
# function from the package after 0.31.1 and it fails for Python 3.6?!
${PIP} install --upgrade pip wheel==0.31.1
${PIP} install --upgrade setuptools
${PIP} install --upgrade cython
${PIP} install --upgrade numpy

# NOW we can install requirements
${PIP} install -r /io/requirements.txt
make -C /io/ PYTHON="${PYTHON}"
${PIP} wheel /io/ -w /io/dist/

# Bundle external shared libraries into the wheels.
for whl in /io/dist/*.whl; do
    if [[ "$whl" =~ "$PYMODULE" ]]; then
        auditwheel repair $whl -w /io/dist/ #repair pmdarima wheel and output to /io/dist
    fi

    rm $whl # remove wheel
done
