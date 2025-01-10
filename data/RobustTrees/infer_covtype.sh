#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Covtype (multiclass)
if [ ! -f predictions/covtype/test_pred.csv ]; then
    mkdir -p predictions/covtype
    pushd repo
    ./xgboost data/covtype.scale01.conf
    mv "$(ls -t | head -n1)" ../predictions/covtype/covtype.model
    popd
    python predict.py --model_path predictions/covtype/covtype.model --data repo/data/covtype.scale01.train0 --output_path predictions/covtype/train_pred.csv
    python predict.py --model_path predictions/covtype/covtype.model --data repo/data/covtype.scale01.test0 --output_path predictions/covtype/test_pred.csv
else
    echo "Skipping covtype as predictions already exist."
fi

if [ ! -f predictions/covtype.unrob/test_pred.csv ]; then
    mkdir -p predictions/covtype.unrob
    pushd repo
    ./xgboost data/covtype.scale01.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/covtype.unrob/covtype.unrob.model
    popd
    python predict.py --model_path predictions/covtype.unrob/covtype.unrob.model --data repo/data/covtype.scale01.train0 --output_path predictions/covtype.unrob/train_pred.csv
    python predict.py --model_path predictions/covtype.unrob/covtype.unrob.model --data repo/data/covtype.scale01.test0 --output_path predictions/covtype.unrob/test_pred.csv
else
    echo "Skipping covtype.unrob as predictions already exist."
fi
