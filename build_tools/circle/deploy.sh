#!/bin/bash

set -e -x

pip install twine wheel
if [[ $CIRCLE_TAG =~ '[a-zA-Z]'$ ]]; then
  twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/pmdarima-*
else
  twine upload --skip-existing dist/pmdarima-*
fi
# twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/pmdarima-*
