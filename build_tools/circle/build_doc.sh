#!/bin/bash

set -e

pip install --upgrade .
make install
pip install pandas sphinx sphinx_gallery pytest-runner sphinx_rtd_theme "matplotlib>=2.2.0" image
pip install --upgrade numpydoc

# this is a hack, but we have to make sure we're only ever running this from
# the top level of the package and not in the subdirectory...
if [[ ! -d pmdarima/__check_build ]]; then
    echo "This must be run from the pmdarima project directory"
    exit 3
fi

make docker-documentation PMDARIMA_VERSION=$(build_tools/get_tag.py)
