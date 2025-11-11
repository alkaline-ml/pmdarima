#!/bin/bash

set -e

pip install --upgrade -r requirements.txt
make install
pip install pandas sphinx sphinx_gallery pytest-runner sphinx_rtd_theme "matplotlib>=2.2.0" image
pip install --upgrade numpydoc

# this is a hack, but we have to make sure we're only ever running this from
# the top level of the package and not in the subdirectory...
if [[ ! -d pmdarima/__check_build ]]; then
    echo "This must be run from the pmdarima project directory"
    exit 3
fi

# Set ${version}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/get_version.sh"

make docker-documentation PMDARIMA_VERSION=${version}
