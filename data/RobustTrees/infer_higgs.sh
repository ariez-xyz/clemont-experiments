#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# HIGGS
if [ ! -f predictions/higgs/test_pred.csv ]; then
    mkdir -p predictions/higgs
    pushd repo
    ./xgboost data/higgs.conf
    mv "$(ls -t | head -n1)" ../predictions/higgs/higgs.model
    popd
    python predict.py --model_path predictions/higgs/higgs.model --data repo/data/HIGGS_s.train0 --output_path predictions/higgs/train_pred.csv --binary
    python predict.py --model_path predictions/higgs/higgs.model --data repo/data/HIGGS_s.test0 --output_path predictions/higgs/test_pred.csv --binary
else
    echo "Skipping higgs as predictions already exist."
fi

if [ ! -f predictions/higgs.unrob/test_pred.csv ]; then
    mkdir -p predictions/higgs.unrob
    pushd repo
    ./xgboost data/higgs.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/higgs.unrob/higgs.unrob.model
    popd
    python predict.py --model_path predictions/higgs.unrob/higgs.unrob.model --data repo/data/HIGGS_s.train0 --output_path predictions/higgs.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/higgs.unrob/higgs.unrob.model --data repo/data/HIGGS_s.test0 --output_path predictions/higgs.unrob/test_pred.csv --binary
else
    echo "Skipping higgs.unrob as predictions already exist."
fi
