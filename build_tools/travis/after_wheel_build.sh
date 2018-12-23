#!/usr/bin/env bash

# Don't think we actually need this, because we always build, but it doesn't hurt
if [[ "${DEPLOY}" != true ]]; then
    # Not a release
    exit 0
fi

# Check for presence of wheel and tarball post-build
WHEEL=$(find dist/ -name '*.whl' | wc -l)
TAR=$(find dist/ -name '*.tar.gz' | wc -l)
if [[ ${WHEEL} -gt 0 && ${TAR} -gt 0 ]]; then
    echo "Build artifacts created successfully"
else
    echo "Build artifacts were not created. Skipping deployment..."
    exit 1
fi
