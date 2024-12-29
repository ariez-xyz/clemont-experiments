#!/bin/bash

mkdir -p predictions  # Create predictions directory if it does not exist

# Robust ijcnn
if [ ! -f predictions/ijcnn/test_pred.csv ]; then
    mkdir predictions/ijcnn
    pushd repo
    ./xgboost data/ijcnn.conf # Train model
    mv "$(ls -t | head -n1)" ../predictions/ijcnn/ijcnn.model # Pick final checkpoint
    popd
    python predict.py --model_path predictions/ijcnn/ijcnn.model --data repo/data/ijcnn1s0.t --output_path predictions/ijcnn/test_pred.csv --binary
else
    echo "Skipping ijcnn as predictions already exist."
fi

# Unrobust ijcnn
if [ ! -f predictions/ijcnn.unrob/test_pred.csv ]; then
    mkdir predictions/ijcnn.unrob
    pushd repo
    ./xgboost data/ijcnn.unrob.conf # Train model
    mv "$(ls -t | head -n1)" ../predictions/ijcnn.unrob/ijcnn.unrob.model # Pick final checkpoint
    popd
    python predict.py --model_path predictions/ijcnn.unrob/ijcnn.unrob.model --data repo/data/ijcnn1s0.t --output_path predictions/ijcnn.unrob/test_pred.csv --binary
else
    echo "Skipping ijcnn.unrob as predictions already exist."
fi
