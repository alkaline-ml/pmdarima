#!/bin/bash

PYTHON=$1

# Set up virtual env
$PYTHON -m pip install virtualenv
$PYTHON -m venv .env
source .env/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r build_tools/build_requirements.txt
pip install -r requirements.txt

# Make our wheel
make version bdist_wheel

# Audit our wheel
pip install auditwheel
for whl in dist/*.whl; do
    if [[ "$whl" =~ "pmdarima" ]]; then
        auditwheel repair $whl -w dist/ #repair pmdarima wheel and output to dist/
    fi
    rm "$whl" # remove original wheel
done

# Testing on aarch64 takes too long, so we simply import the package as a spot test
pip install --pre --no-index --find-links dist/ pmdarima
cd .github # Can't be in the top-level directory for import or it will fail
$PYTHON -c 'import pmdarima; pmdarima.show_versions()'
