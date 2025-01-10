#!/bin/bash

MODEL=${1:-Standard}
DATASET=${2:-cifar10}

mkdir -p predictions

python predict.py --model "$MODEL" --dataset "$DATASET" --output "predictions/$DATASET-$MODEL.csv"

