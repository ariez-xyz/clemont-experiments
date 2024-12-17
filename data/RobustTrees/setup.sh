#!/bin/bash

git clone --recursive https://github.com/chenhongge/RobustTrees repo
pushd repo
./build.sh
pushd data
./download_data.sh
popd
