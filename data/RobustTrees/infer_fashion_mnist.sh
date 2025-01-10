#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Fashion MNIST (multiclass)
if [ ! -f predictions/fashion/test_pred.csv ]; then
    mkdir -p predictions/fashion
    pushd repo
    ./xgboost data/fashion.conf
    mv "$(ls -t | head -n1)" ../predictions/fashion/fashion.model
    popd
    python predict.py --model_path predictions/fashion/fashion.model --data repo/data/fashion.train0 --output_path predictions/fashion/train_pred.csv
    python predict.py --model_path predictions/fashion/fashion.model --data repo/data/fashion.test0 --output_path predictions/fashion/test_pred.csv
else
    echo "Skipping fashion as predictions already exist."
fi

if [ ! -f predictions/fashion.unrob/test_pred.csv ]; then
    mkdir -p predictions/fashion.unrob
    pushd repo
    ./xgboost data/fashion.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/fashion.unrob/fashion.unrob.model
    popd
    python predict.py --model_path predictions/fashion.unrob/fashion.unrob.model --data repo/data/fashion.train0 --output_path predictions/fashion.unrob/train_pred.csv
    python predict.py --model_path predictions/fashion.unrob/fashion.unrob.model --data repo/data/fashion.test0 --output_path predictions/fashion.unrob/test_pred.csv
else
    echo "Skipping fashion.unrob as predictions already exist."
fi

