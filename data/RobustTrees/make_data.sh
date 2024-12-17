#!/bin/bash

# TODO: 

# Robust ijcnn
if [ ! -f predictions/ijcnn/test_pred.csv ]; then
    mkdir predictions/ijcnn
    pushd repo
    ./xgboost data/ijcnn.conf # Train model
    mv "$(ls -t | head -n1)" ../predictions/ijcnn/ijcnn.model # Pick final checkpoint
    popd
    python predict.py -d=repo/data/ijcnn1s0.t -m predictions/ijcnn/ijcnn.model -f predictions/ijcnn/test_pred.csv -c 2  --feature_start=0 # Save predictions as csv
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
	python predict.py -d=repo/data/ijcnn1s0.t -m predictions/ijcnn.unrob/ijcnn.unrob.model -f predictions/ijcnn.unrob/test_pred.csv -c 2  --feature_start=0 # Save predictions as csv
else
    echo "Skipping ijcnn.unrob as predictions already exist."
fi
