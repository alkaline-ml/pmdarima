#!/bin/bash

# The version is retrieved from the CIRCLE_TAG. If there is no version, we just
# call it 0.0.0, since we won't be pushing anyways (not master and no tag)
if [[ -n ${CIRCLE_TAG} ]]; then
    # We should have the VERSION file on tags now since 'make documentation'
    # gets it. If not, we use 0.0.0. There are two cases we ever deploy:
    #   1. Master (where version is not used, as we use 'develop'
    #   2. Tags (where version IS defined)
    echo "On tag"
    make version
    export version=$(cat pmdarima/VERSION)
else
    echo "Not on tag, will use version=0.0.0"
    export version="0.0.0"
fi
