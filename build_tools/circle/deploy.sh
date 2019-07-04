#!/bin/bash

set -e -x

pip install twine wheel
# twine upload dist/pmdarima-*
twine upload --repository-url https://test.pypi.org/legacy/ dist/pmdarima-*
