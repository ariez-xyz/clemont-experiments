#!/bin/bash

mkdir -p predictions

# IJCNN
if [ ! -f predictions/ijcnn/test_pred.csv ]; then
    mkdir predictions/ijcnn
    pushd repo
    ./xgboost data/ijcnn.conf
    mv "$(ls -t | head -n1)" ../predictions/ijcnn/ijcnn.model
    popd
    python predict.py --model_path predictions/ijcnn/ijcnn.model --data repo/data/ijcnn1s0 --output_path predictions/ijcnn/train_pred.csv --binary
    python predict.py --model_path predictions/ijcnn/ijcnn.model --data repo/data/ijcnn1s0.t --output_path predictions/ijcnn/test_pred.csv --binary
else
    echo "Skipping ijcnn as predictions already exist."
fi

if [ ! -f predictions/ijcnn.unrob/test_pred.csv ]; then
    mkdir predictions/ijcnn.unrob
    pushd repo
    ./xgboost data/ijcnn.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/ijcnn.unrob/ijcnn.unrob.model
    popd
    python predict.py --model_path predictions/ijcnn.unrob/ijcnn.unrob.model --data repo/data/ijcnn1s0 --output_path predictions/ijcnn.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/ijcnn.unrob/ijcnn.unrob.model --data repo/data/ijcnn1s0.t --output_path predictions/ijcnn.unrob/test_pred.csv --binary
else
    echo "Skipping ijcnn.unrob as predictions already exist."
fi

## TODO: Very large, will do later.
## HIGGS
#if [ ! -f predictions/higgs/test_pred.csv ]; then
#    mkdir predictions/higgs
#    pushd repo
#    ./xgboost data/higgs.conf
#    mv "$(ls -t | head -n1)" ../predictions/higgs/higgs.model
#    popd
#    python predict.py --model_path predictions/higgs/higgs.model --data repo/data/HIGGS_s.train0 --output_path predictions/higgs/train_pred.csv --binary
#    python predict.py --model_path predictions/higgs/higgs.model --data repo/data/HIGGS_s.test0 --output_path predictions/higgs/test_pred.csv --binary
#else
#    echo "Skipping higgs as predictions already exist."
#fi
#
#if [ ! -f predictions/higgs.unrob/test_pred.csv ]; then
#    mkdir predictions/higgs.unrob
#    pushd repo
#    ./xgboost data/higgs.unrob.conf
#    mv "$(ls -t | head -n1)" ../predictions/higgs.unrob/higgs.unrob.model
#    popd
#    python predict.py --model_path predictions/higgs.unrob/higgs.unrob.model --data repo/data/HIGGS_s.train0 --output_path predictions/higgs.unrob/train_pred.csv --binary
#    python predict.py --model_path predictions/higgs.unrob/higgs.unrob.model --data repo/data/HIGGS_s.test0 --output_path predictions/higgs.unrob/test_pred.csv --binary
#else
#    echo "Skipping higgs.unrob as predictions already exist."
#fi

## TODO: Throws an error?
## Binary MNIST
#if [ ! -f predictions/binary_mnist/test_pred.csv ]; then
#    mkdir predictions/binary_mnist
#    pushd repo
#    ./xgboost data/binary_mnist.conf
#    mv "$(ls -t | head -n1)" ../predictions/binary_mnist/binary_mnist.model
#    popd
#    python predict.py --model_path predictions/binary_mnist/binary_mnist.model --data repo/data/binary_mnist0 --output_path predictions/binary_mnist/train_pred.csv --binary
#    python predict.py --model_path predictions/binary_mnist/binary_mnist.model --data repo/data/binary_mnist0.t --output_path predictions/binary_mnist/test_pred.csv --binary
#else
#    echo "Skipping binary_mnist as predictions already exist."
#fi
#
#if [ ! -f predictions/binary_mnist.unrob/test_pred.csv ]; then
#    mkdir predictions/binary_mnist.unrob
#    pushd repo
#    ./xgboost data/binary_mnist.unrob.conf
#    mv "$(ls -t | head -n1)" ../predictions/binary_mnist.unrob/binary_mnist.unrob.model
#    popd
#    python predict.py --model_path predictions/binary_mnist.unrob/binary_mnist.unrob.model --data repo/data/binary_mnist0 --output_path predictions/binary_mnist.unrob/train_pred.csv --binary
#    python predict.py --model_path predictions/binary_mnist.unrob/binary_mnist.unrob.model --data repo/data/binary_mnist0.t --output_path predictions/binary_mnist.unrob/test_pred.csv --binary
#else
#    echo "Skipping binary_mnist.unrob as predictions already exist."
#fi

# Original MNIST (multiclass)
if [ ! -f predictions/ori_mnist/test_pred.csv ]; then
    mkdir predictions/ori_mnist
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
    mkdir predictions/ori_mnist.unrob
    pushd repo
    ./xgboost data/ori_mnist.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/ori_mnist.unrob/ori_mnist.unrob.model
    popd
    python predict.py --model_path predictions/ori_mnist.unrob/ori_mnist.unrob.model --data repo/data/ori_mnist.train0 --output_path predictions/ori_mnist.unrob/train_pred.csv
    python predict.py --model_path predictions/ori_mnist.unrob/ori_mnist.unrob.model --data repo/data/ori_mnist.test0 --output_path predictions/ori_mnist.unrob/test_pred.csv
else
    echo "Skipping ori_mnist.unrob as predictions already exist."
fi

# Fashion MNIST (multiclass)
if [ ! -f predictions/fashion/test_pred.csv ]; then
    mkdir predictions/fashion
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
    mkdir predictions/fashion.unrob
    pushd repo
    ./xgboost data/fashion.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/fashion.unrob/fashion.unrob.model
    popd
    python predict.py --model_path predictions/fashion.unrob/fashion.unrob.model --data repo/data/fashion.train0 --output_path predictions/fashion.unrob/train_pred.csv
    python predict.py --model_path predictions/fashion.unrob/fashion.unrob.model --data repo/data/fashion.test0 --output_path predictions/fashion.unrob/test_pred.csv
else
    echo "Skipping fashion.unrob as predictions already exist."
fi

# Covtype (multiclass)
if [ ! -f predictions/covtype/test_pred.csv ]; then
    mkdir predictions/covtype
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
    mkdir predictions/covtype.unrob
    pushd repo
    ./xgboost data/covtype.scale01.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/covtype.unrob/covtype.unrob.model
    popd
    python predict.py --model_path predictions/covtype.unrob/covtype.unrob.model --data repo/data/covtype.scale01.train0 --output_path predictions/covtype.unrob/train_pred.csv
    python predict.py --model_path predictions/covtype.unrob/covtype.unrob.model --data repo/data/covtype.scale01.test0 --output_path predictions/covtype.unrob/test_pred.csv
else
    echo "Skipping covtype.unrob as predictions already exist."
fi

# Sensorless (multiclass)
if [ ! -f predictions/sensorless/test_pred.csv ]; then
    mkdir predictions/sensorless
    pushd repo
    ./xgboost data/Sensorless.conf
    mv "$(ls -t | head -n1)" ../predictions/sensorless/sensorless.model
    popd
    python predict.py --model_path predictions/sensorless/sensorless.model --data repo/data/Sensorless.scale.tr0 --output_path predictions/sensorless/train_pred.csv
    python predict.py --model_path predictions/sensorless/sensorless.model --data repo/data/Sensorless.scale.val0 --output_path predictions/sensorless/test_pred.csv
else
    echo "Skipping sensorless as predictions already exist."
fi

if [ ! -f predictions/sensorless.unrob/test_pred.csv ]; then
    mkdir predictions/sensorless.unrob
    pushd repo
    ./xgboost data/Sensorless.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/sensorless.unrob/sensorless.unrob.model
    popd
    python predict.py --model_path predictions/sensorless.unrob/sensorless.unrob.model --data repo/data/Sensorless.scale.tr0 --output_path predictions/sensorless.unrob/train_pred.csv
    python predict.py --model_path predictions/sensorless.unrob/sensorless.unrob.model --data repo/data/Sensorless.scale.val0 --output_path predictions/sensorless.unrob/test_pred.csv
else
    echo "Skipping sensorless.unrob as predictions already exist."
fi

# Breast Cancer
if [ ! -f predictions/breast_cancer/test_pred.csv ]; then
    mkdir predictions/breast_cancer
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
    mkdir predictions/breast_cancer.unrob
    pushd repo
    ./xgboost data/breast_cancer.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/breast_cancer.unrob/breast_cancer.unrob.model
    popd
    python predict.py --model_path predictions/breast_cancer.unrob/breast_cancer.unrob.model --data repo/data/breast_cancer_scale0.train --output_path predictions/breast_cancer.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/breast_cancer.unrob/breast_cancer.unrob.model --data repo/data/breast_cancer_scale0.test --output_path predictions/breast_cancer.unrob/test_pred.csv --binary
else
    echo "Skipping breast_cancer.unrob as predictions already exist."
fi

# Cod-RNA
if [ ! -f predictions/cod-rna/test_pred.csv ]; then
    mkdir predictions/cod-rna
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
    mkdir predictions/cod-rna.unrob
    pushd repo
    ./xgboost data/cod-rna.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/cod-rna.unrob/cod-rna.unrob.model
    popd
    python predict.py --model_path predictions/cod-rna.unrob/cod-rna.unrob.model --data repo/data/cod-rna_s0 --output_path predictions/cod-rna.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/cod-rna.unrob/cod-rna.unrob.model --data repo/data/cod-rna_s.t0 --output_path predictions/cod-rna.unrob/test_pred.csv --binary
else
    echo "Skipping cod-rna.unrob as predictions already exist."
fi

# Diabetes
if [ ! -f predictions/diabetes/test_pred.csv ]; then
    mkdir predictions/diabetes
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
    mkdir predictions/diabetes.unrob
    pushd repo
    ./xgboost data/diabetes.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/diabetes.unrob/diabetes.unrob.model
    popd
    python predict.py --model_path predictions/diabetes.unrob/diabetes.unrob.model --data repo/data/diabetes_scale0.train --output_path predictions/diabetes.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/diabetes.unrob/diabetes.unrob.model --data repo/data/diabetes_scale0.test --output_path predictions/diabetes.unrob/test_pred.csv --binary
else
    echo "Skipping diabetes.unrob as predictions already exist."
fi

# Webspam
if [ ! -f predictions/webspam/test_pred.csv ]; then
    mkdir predictions/webspam
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
    mkdir predictions/webspam.unrob
    pushd repo
    ./xgboost data/webspam.unrob.conf
    mv "$(ls -t | head -n1)" ../predictions/webspam.unrob/webspam.unrob.model
    popd
    python predict.py --model_path predictions/webspam.unrob/webspam.unrob.model --data repo/data/webspam_wc_normalized_unigram.svm0.train --output_path predictions/webspam.unrob/train_pred.csv --binary
    python predict.py --model_path predictions/webspam.unrob/webspam.unrob.model --data repo/data/webspam_wc_normalized_unigram.svm0.test --output_path predictions/webspam.unrob/test_pred.csv --binary
else
    echo "Skipping webspam.unrob as predictions already exist."
fi
