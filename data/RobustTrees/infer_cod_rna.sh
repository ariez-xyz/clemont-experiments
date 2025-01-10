#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Cod-RNA
if [ ! -f predictions/cod-rna/test_pred.csv ]; then
    mkdir -p predictions/cod-rna
    pushd repo
    ./xgboost data/cod-rna.conf
    mv "$(ls -t | head -n1)" ../predictions/cod-rna/cod-rna.model
    popd
    python predict.py --model_path predictions/cod-rna/cod-rna.model --data repo/data/cod-rna_s0 --output_path predictions/cod-rna/train_pred.csv --binary
    python predict.py --model_path predictions/cod-rna/cod-rna.model --data repo/data/cod-rna_s.t0 --output_path predictions/cod-rna/test_pred.csv --binary
else
    echo "Skipping cod-rna as predictions already exist."
fi

if [ ! -f predictions/cod-rna.unrob/test_pred.csv ]; then
    mkdir -p predictions/cod-rna.unrob
    pushd repo
    ./xgboost data/cod-rna.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/cod-rna.unrob/cod-rna.unrob.model
    popd
    python predict.py --model_path predictions/cod-rna.unrob/cod-rna.unrob.model --data repo/data/cod-rna_s0 --output_path predictions/cod-rna.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/cod-rna.unrob/cod-rna.unrob.model --data repo/data/cod-rna_s.t0 --output_path predictions/cod-rna.unrob/test_pred.csv --binary
else
    echo "Skipping cod-rna.unrob as predictions already exist."
fi
