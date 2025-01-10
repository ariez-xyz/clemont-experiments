#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Breast Cancer
if [ ! -f predictions/breast_cancer/test_pred.csv ]; then
    mkdir -p predictions/breast_cancer
    pushd repo
    ./xgboost data/breast_cancer.conf
    mv "$(ls -t | head -n1)" ../predictions/breast_cancer/breast_cancer.model
    popd
    python predict.py --model_path predictions/breast_cancer/breast_cancer.model --data repo/data/breast_cancer_scale0.train --output_path predictions/breast_cancer/train_pred.csv --binary
    python predict.py --model_path predictions/breast_cancer/breast_cancer.model --data repo/data/breast_cancer_scale0.test --output_path predictions/breast_cancer/test_pred.csv --binary
else
    echo "Skipping breast_cancer as predictions already exist."
fi

if [ ! -f predictions/breast_cancer.unrob/test_pred.csv ]; then
    mkdir -p predictions/breast_cancer.unrob
    pushd repo
    ./xgboost data/breast_cancer.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/breast_cancer.unrob/breast_cancer.unrob.model
    popd
    python predict.py --model_path predictions/breast_cancer.unrob/breast_cancer.unrob.model --data repo/data/breast_cancer_scale0.train --output_path predictions/breast_cancer.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/breast_cancer.unrob/breast_cancer.unrob.model --data repo/data/breast_cancer_scale0.test --output_path predictions/breast_cancer.unrob/test_pred.csv --binary
else
    echo "Skipping breast_cancer.unrob as predictions already exist."
fi
