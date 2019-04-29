#!/bin/bash

if python -c "from twine.commands.check import check; check(['dist/*'])" | grep  "warning"; then
    echo "README will not render properly on PyPI"
    exit 1
else
    echo "README rendered appropriately"
fi