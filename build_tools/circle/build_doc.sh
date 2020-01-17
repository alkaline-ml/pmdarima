#!/bin/bash

# TODO: what if it's a tag? Will it fail since the version dir already exists?

set -e

# this is a hack, but we have to make sure we're only ever running this from
# the top level of the package and not in the subdirectory...
if [[ ! -d pmdarima/__check_build ]]; then
    echo "This must be run from the pmdarima project directory"
    exit 3
fi

# The version is retrieved from the CIRCLE_TAG. If there is no version, we just
# call it 0.0.0, since we won't be pushing anyways (not master and no tag)
if [[ -n ${CIRCLE_TAG} ]]; then
    # We should have the VERSION file on tags now since 'make documentation'
    # gets it. If not, we use 0.0.0. There are two cases we ever deploy:
    #   1. Master (where version is not used, as we use 'develop'
    #   2. Tags (where version IS defined)
    echo "On tag"
    make version
    version=$(cat pmdarima/VERSION)
else
    echo "Not on tag, will use version=0.0.0"
    version="0.0.0"
fi

# get the running branch
# branch=$(git symbolic-ref --short HEAD)

# cd into docs, make them
# cd doc
# make clean html EXAMPLES_PATTERN=example_*
# cd ..
make documentation PMDARIMA_VERSION=${version}
