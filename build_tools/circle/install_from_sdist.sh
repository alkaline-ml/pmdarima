#!/bin/bash

set -e

make sdist

# it will always be 0.0.0 since we didn't version it
cd dist
pip install pmdarima-0.0.0.tar.gz
python -c 'import pmdarima as pm'
