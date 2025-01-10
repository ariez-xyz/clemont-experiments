#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Diabetes
if [ ! -f predictions/diabetes/test_pred.csv ]; then
    mkdir -p predictions/diabetes
    pushd repo
    ./xgboost data/diabetes.conf
    mv "$(ls -t | head -n1)" ../predictions/diabetes/diabetes.model
    popd
    python predict.py --model_path predictions/diabetes/diabetes.model --data repo/data/diabetes_scale0.train --output_path predictions/diabetes/train_pred.csv --binary
    python predict.py --model_path predictions/diabetes/diabetes.model --data repo/data/diabetes_scale0.test --output_path predictions/diabetes/test_pred.csv --binary
else
    echo "Skipping diabetes as predictions already exist."
fi

if [ ! -f predictions/diabetes.unrob/test_pred.csv ]; then
    mkdir -p predictions/diabetes.unrob
    pushd repo
    ./xgboost data/diabetes.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/diabetes.unrob/diabetes.unrob.model
    popd
    python predict.py --model_path predictions/diabetes.unrob/diabetes.unrob.model --data repo/data/diabetes_scale0.train --output_path predictions/diabetes.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/diabetes.unrob/diabetes.unrob.model --data repo/data/diabetes_scale0.test --output_path predictions/diabetes.unrob/test_pred.csv --binary
else
    echo "Skipping diabetes.unrob as predictions already exist."
fi
