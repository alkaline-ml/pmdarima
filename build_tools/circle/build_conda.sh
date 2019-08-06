#!/bin/bash

pip install conda jinja2
mkdir conda
ls
cd build_tools/common
python render_meta.py
cd -
conda install conda-build
conda-build --python=$1 conda/
