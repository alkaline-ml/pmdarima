#!/bin/bash

set -e

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
