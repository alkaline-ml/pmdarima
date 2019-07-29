#!/bin/bash

# change the package name to the existing PyPi package you would like to build and adjust the Python versions
pkg='pmdarima'
array=( 3.5 3.6 3.7 )

echo "Building conda package ..."
cd $pkg
wget https://raw.githubusercontent.com/conda/conda/master/conda.recipe/build.sh
wget https://raw.githubusercontent.com/conda/conda/master/conda.recipe/bld.bat
cd ~

# building conda packages
for i in "${array[@]}"
do
	conda build --python $i $pkg
done

# convert package to other platforms
cd ~
platforms=( osx-64 linux-32 linux-64 win-32 win-64 )
find $HOME/conda-bld/linux-64/ -name *.tar.bz2 | while read file
do
    echo $file
    #conda convert --platform all $file  -o $HOME/conda-bld/
    for platform in "${platforms[@]}"
    do
       conda convert --platform $platform $file  -o $HOME/conda-bld/
    done

done

# upload packages to conda
#find $HOME/conda-bld/ -name *.tar.bz2 | while read file
#do
#    echo $file
#    anaconda upload $file
#done

echo "Building conda package done!"