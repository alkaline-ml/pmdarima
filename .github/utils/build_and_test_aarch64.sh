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

# Install and run tests
pip install --pre --no-index --find-links dist/ pmdarima
pytest --showlocals --durations=20 --pyargs pmdarima --benchmark-skip
