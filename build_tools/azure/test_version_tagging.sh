#!/bin/bash

set -e
pip install pathlib

BUILD_SOURCEBRANCH=refs/tags/v0.99.999 python ${BUILD_SOURCESDIRECTORY}/build_tools/get_tag.py

if [[ $(cat ${BUILD_SOURCESDIRECTORY}/pmdarima/VERSION) =~ '^[0-9]+\.[0-9]+\.?[0-9]*?[a-zA-Z]+[0-9]*$' ]]; then
  # Adding the `test` label means it doesn't show up unless you specifically
  # search for packages with the label `test`
  echo 'Uploading to conda with test label'
  anaconda upload --label test --skip $output_file
elif [[ $(cat ${BUILD_SOURCESDIRECTORY}/pmdarima/VERSION) =~ '^[0-9]+\.[0-9]+\.?[0-9]*?$' ]]; then
  echo 'Uploading to production conda channel'
  anaconda upload --skip $output_file
else
  echo "Malformed tag: $(cat ${BUILD_SOURCESDIRECTORY}/pmdarima/VERSION)"
  exit 1
fi

if [[ ! -f ${BUILD_SOURCESDIRECTORY}/pmdarima/VERSION ]]; then
    echo "Expected VERSION file"
    exit 4
fi
