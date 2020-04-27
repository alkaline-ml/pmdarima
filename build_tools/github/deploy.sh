#!/bin/bash

if [[ $(cat ${GITHUB_WORKSPACE}/pmdarima/VERSION) =~ ^[0-9]+\.[0-9]+\.?[0-9]*[a-zA-Z]+[0-9]*$ ]]; then
  echo 'Uploading to test pypi'
  python -m twine upload --repository-url https://test.pypi.org/legacy/ --skip-existing dist/pmdarima-*
elif [[ $(cat ${GITHUB_WORKSPACE}/pmdarima/VERSION) =~ ^[0-9]+\.[0-9]+\.?[0-9]*$ ]]; then
  echo 'Uploading to production pypi'
  python -m twine upload --skip-existing dist/pmdarima-*
else
  echo "Malformed tag: $(cat ${GITHUB_WORKSPACE}/pmdarima/VERSION)"
  exit 1
fi