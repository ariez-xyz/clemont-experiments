#!/bin/bash

pushd ../../
source activate.sh
popd

pushd repo
micromamba activate lcifr
source setup.sh
cd code/experiments/
./noise.sh
