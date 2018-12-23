#!/usr/bin/env bash

# Don't think we actually need this, because we always build, but it doesn't hurt
if [[ "${DEPLOY}" != true ]]; then
    # Not a release
    exit 0
fi

# Check for presence of wheel and tarball post-build
if [[ ! -f dist/*.whl && ! -f dist/*.tar.gz ]]; then
    echo "Artifacts did not build successfully"
    exit 1
else
    echo "Build artifacts created successfully"
fi
