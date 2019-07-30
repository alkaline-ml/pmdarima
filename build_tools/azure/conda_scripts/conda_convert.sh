#!/bin/bash

# convert package to other platforms
#platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
find $HOME/conda-bld/linux-64/ -name *.tar.bz2 | while read file
do
    echo $file
    conda convert --platform all $file  -o $HOME/conda-bld/
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
