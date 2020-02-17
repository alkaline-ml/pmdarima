#!/bin/bash

# This is run from the root directory, so we treat it as such

echo "In build.sh"

echo "PYTHON: $(which python)"

# Install our dependencies
python -m pip install -r requirements.txt

# Build our distribution first
python -m pip install --no-deps --ignore-installed .

echo "Installed pmdarima"

# Find where the package was installed
site_packages=$(python -c 'import site; print(site.getsitepackages()[0])')

echo "Found site_packages: ${site_packages}"

# Make the output directory
output_dir=${PREFIX}/lib/python${PY_VER}/site-packages/
mkdir $output_dir

echo "Made output_dir: ${output_dir}"

# Copy the build artifacts to the conda output directory
cp ${site_packages}/${PKG_NAME} $output_dir
cp ${site_packages}/${PKG_NAME}-${PKG_VERSION}.dist-info $output_dir
