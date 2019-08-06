#!/bin/bash

# Build the wheel first
pip install -r requirements.txt
python setup.py bdist_wheel

# Render the meta file
pip install jinja2
mkdir conda
cd build_tools/common
python render_meta.py
cd -

# Build the conda distribution
conda install conda-build
conda-build --python=$1 conda/
