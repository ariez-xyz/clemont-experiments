#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Binary MNIST
if [ ! -f predictions/binary_mnist/test_pred.csv ]; then
    mkdir -p predictions/binary_mnist
    pushd repo
    ./xgboost data/binary_mnist.conf
    mv "$(ls -t | head -n1)" ../predictions/binary_mnist/binary_mnist.model
    popd
    python predict.py --model_path predictions/binary_mnist/binary_mnist.model --data repo/data/binary_mnist0 --output_path predictions/binary_mnist/train_pred.csv --binary
    python predict.py --model_path predictions/binary_mnist/binary_mnist.model --data repo/data/binary_mnist0.t --output_path predictions/binary_mnist/test_pred.csv --binary
else
    echo "Skipping binary_mnist as predictions already exist."
fi

if [ ! -f predictions/binary_mnist.unrob/test_pred.csv ]; then
    mkdir -p predictions/binary_mnist.unrob
    pushd repo
    ./xgboost data/binary_mnist.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/binary_mnist.unrob/binary_mnist.unrob.model
    popd
    python predict.py --model_path predictions/binary_mnist.unrob/binary_mnist.unrob.model --data repo/data/binary_mnist0 --output_path predictions/binary_mnist.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/binary_mnist.unrob/binary_mnist.unrob.model --data repo/data/binary_mnist0.t --output_path predictions/binary_mnist.unrob/test_pred.csv --binary
else
    echo "Skipping binary_mnist.unrob as predictions already exist."
fi

