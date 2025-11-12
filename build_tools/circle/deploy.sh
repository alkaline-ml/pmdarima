#!/bin/bash

set -e -x

pip install twine wheel

# Check our VERSION. Basically, if it contains letters, it is a pre-release. Otherwise,
# it has to match X.Y or X.Y.Z
#
# On CircleCI, we look for the `v` at the beginning of the version, since we are looking at the tag
if [[ ${CIRCLE_TAG} =~ ^v?[0-9]+\.[0-9]+\.?[0-9]*[a-zA-Z]+[0-9]*$ ]]; then
  echo 'Uploading to test pypi'
  TWINE_PASSWORD=$TEST_PYPI_PASSWORD twine upload --skip-existing --repository-url https://test.pypi.org/legacy/ dist/pmdarima-*
elif [[ ${CIRCLE_TAG} =~ ^v?[0-9]+\.[0-9]+\.?[0-9]*$ ]]; then
  echo 'Uploading to production pypi'
  twine upload --skip-existing dist/pmdarima-*
else
  echo "Malformed tag: ${CIRCLE_TAG}"
  exit 1
fi
