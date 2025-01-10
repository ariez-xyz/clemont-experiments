#!/bin/bash
pushd ../../
source activate.sh
popd
pip install git+https://github.com/RobustBench/robustbench.git
