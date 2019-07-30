#!/bin/bash

# Adapted from this gist: https://gist.github.com/giswqs/4eb62fb08658c8a200c4e18bb5e6270c

# convert package to other platforms
#platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
find $CONDA/conda-bld/linux-64/ -name *.tar.bz2 | while read file
do
    echo $file
    conda convert -f --platform all $file -o $CONDA/conda-bld/
#    for platform in "${platforms[@]}"
#    do
#       conda convert --platform $platform $file  -o $HOME/conda-bld/
#    done

done

# upload packages to conda
#find $HOME/conda-bld/ -name *.tar.bz2 | while read file
#do
#    echo $file
#    anaconda upload $file
#done
