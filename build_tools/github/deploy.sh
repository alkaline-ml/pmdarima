#!/bin/bash

# Extract version from git tag
VERSION=${GITHUB_REF#refs/tags/v}

if [[ $VERSION =~ ^[0-9]+\.[0-9]+\.?[0-9]*[a-zA-Z]+[0-9]*$ ]]; then
  echo "Pre-release version: $VERSION - Uploading to test pypi"
  TWINE_PASSWORD=$TEST_PYPI_PASSWORD python -m twine upload --repository-url https://test.pypi.org/legacy/ --skip-existing dist/pmdarima-*
elif [[ $VERSION =~ ^[0-9]+\.[0-9]+\.?[0-9]*$ ]]; then
  echo "Production release version: $VERSION - Uploading to production pypi"
  python -m twine upload --skip-existing dist/pmdarima-*
else
  echo "Malformed tag: $VERSION"
  exit 1
fi
