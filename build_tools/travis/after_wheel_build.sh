#!/usr/bin/env bash

# Check for presence of wheel and tarball post-build
WHEEL=$(find dist/ -name '*.whl' | wc -l)
TAR=$(find dist/ -name '*.tar.gz' | wc -l)
if [[ ${WHEEL} -gt 0 && ${TAR} -gt 0 ]]; then
    echo "Build artifacts created successfully"
else
    echo "Build artifacts were not created. Skipping deployment..."
    exit 1
fi

# If we get to this point, the build worked, and we need to check for README rendering
#if python -c "from twine.commands.check import check; check(['dist/*'])" | grep  "warning"; then
#    echo "README will not render properly on PyPI"
#    exit 1
#else
#    echo "README rendered appropriately"
#fi