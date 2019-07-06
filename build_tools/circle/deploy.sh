#!/bin/bash

set -e -x

pip install twine wheel
twine upload --skip-existing dist/pmdarima-*
# twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/pmdarima-*
