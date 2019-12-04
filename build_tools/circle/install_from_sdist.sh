#!/bin/bash

set -e

python setup.py sdist

# we'll need numpy and cython to build this. let install_requires do all the
# rest of the work.
pip install "numpy>=1.16" "cython>=0.29"

# it will always be 0.0.0 since we didn't version it
cd dist
pip install pmdarima-0.0.0.tar.gz
python -c 'import pmdarima as pm'
