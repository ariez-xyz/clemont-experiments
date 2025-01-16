#!/bin/bash

MODEL=${1:-Standard}
DATASET=${2:-cifar10}
N=${3:-100}

mkdir -p predictions

python predict.py --model "$MODEL" --dataset "$DATASET" --output "predictions/$DATASET-$MODEL.csv" --n-examples "$N" --threat-model Linf --emb-model small --force-resize

