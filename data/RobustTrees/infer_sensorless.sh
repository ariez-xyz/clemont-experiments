#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Sensorless (multiclass)
if [ ! -f predictions/sensorless/test_pred.csv ]; then
    mkdir -p predictions/sensorless
    pushd repo
    ./xgboost data/Sensorless.conf
    mv "$(ls -t | head -n1)" ../predictions/sensorless/sensorless.model
    popd
    python predict.py --model_path predictions/sensorless/sensorless.model --data repo/data/Sensorless.scale.tr0 --output_path predictions/sensorless/train_pred.csv
    python predict.py --model_path predictions/sensorless/sensorless.model --data repo/data/Sensorless.scale.val0 --output_path predictions/sensorless/test_pred.csv
else
    echo "Skipping sensorless as predictions already exist."
fi

if [ ! -f predictions/sensorless.unrob/test_pred.csv ]; then
    mkdir -p predictions/sensorless.unrob
    pushd repo
    ./xgboost data/Sensorless.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/sensorless.unrob/sensorless.unrob.model
    popd
    python predict.py --model_path predictions/sensorless.unrob/sensorless.unrob.model --data repo/data/Sensorless.scale.tr0 --output_path predictions/sensorless.unrob/train_pred.csv
    python predict.py --model_path predictions/sensorless.unrob/sensorless.unrob.model --data repo/data/Sensorless.scale.val0 --output_path predictions/sensorless.unrob/test_pred.csv
else
    echo "Skipping sensorless.unrob as predictions already exist."
fi

