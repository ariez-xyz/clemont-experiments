#!/bin/bash

pushd ../..
source activate.sh
popd

mkdir -p predictions

# Original MNIST (multiclass)
if [ ! -f predictions/ori_mnist/test_pred.csv ]; then
    mkdir -p predictions/ori_mnist
    pushd repo
    ./xgboost data/ori_mnist.conf
    mv "$(ls -t | head -n1)" ../predictions/ori_mnist/ori_mnist.model
    popd
    python predict.py --model_path predictions/ori_mnist/ori_mnist.model --data repo/data/ori_mnist.train0 --output_path predictions/ori_mnist/train_pred.csv
    python predict.py --model_path predictions/ori_mnist/ori_mnist.model --data repo/data/ori_mnist.test0 --output_path predictions/ori_mnist/test_pred.csv
else
    echo "Skipping ori_mnist as predictions already exist."
fi

if [ ! -f predictions/ori_mnist.unrob/test_pred.csv ]; then
    mkdir -p predictions/ori_mnist.unrob
    pushd repo
    ./xgboost data/ori_mnist.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/ori_mnist.unrob/ori_mnist.unrob.model
    popd
    python predict.py --model_path predictions/ori_mnist.unrob/ori_mnist.unrob.model --data repo/data/ori_mnist.train0 --output_path predictions/ori_mnist.unrob/train_pred.csv
    python predict.py --model_path predictions/ori_mnist.unrob/ori_mnist.unrob.model --data repo/data/ori_mnist.test0 --output_path predictions/ori_mnist.unrob/test_pred.csv
else
    echo "Skipping ori_mnist.unrob as predictions already exist."
fi
