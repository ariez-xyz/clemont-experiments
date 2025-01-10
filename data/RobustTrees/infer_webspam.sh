#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Webspam
if [ ! -f predictions/webspam/test_pred.csv ]; then
    mkdir -p predictions/webspam
    pushd repo
    ./xgboost data/webspam.conf
    mv "$(ls -t | head -n1)" ../predictions/webspam/webspam.model
    popd
    python predict.py --model_path predictions/webspam/webspam.model --data repo/data/webspam_wc_normalized_unigram.svm0.train --output_path predictions/webspam/train_pred.csv --binary
    python predict.py --model_path predictions/webspam/webspam.model --data repo/data/webspam_wc_normalized_unigram.svm0.test --output_path predictions/webspam/test_pred.csv --binary
else
    echo "Skipping webspam as predictions already exist."
fi

if [ ! -f predictions/webspam.unrob/test_pred.csv ]; then
    mkdir -p predictions/webspam.unrob
    pushd repo
    ./xgboost data/webspam.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/webspam.unrob/webspam.unrob.model
    popd
    python predict.py --model_path predictions/webspam.unrob/webspam.unrob.model --data repo/data/webspam_wc_normalized_unigram.svm0.train --output_path predictions/webspam.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/webspam.unrob/webspam.unrob.model --data repo/data/webspam_wc_normalized_unigram.svm0.test --output_path predictions/webspam.unrob/test_pred.csv --binary
else
    echo "Skipping webspam.unrob as predictions already exist."
fi
