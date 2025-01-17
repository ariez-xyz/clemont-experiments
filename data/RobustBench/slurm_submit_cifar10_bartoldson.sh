#!/bin/bash
#
# To submit: sbatch script.sh
#
# IMPORTANT: This section breaks when placed towards the end of the config options
# -------------------------------------------------------
# Use GPU partition, use one GPU, specify GPU type
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
# -------------------------------------------------------
#
#SBATCH --job-name=bartoldson-cifar10
#SBATCH --output=bartoldson-cifar10-%j.log
#
# 8 cores
#SBATCH -c 8
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#
# Do not requeue the job in the case it fails.
#SBATCH --no-requeue
# Export the local environment to the compute nodes
#SBATCH --export=ALL
#

MODEL=Bartoldson2024Adversarial_WRN-94-16
DATASET=cifar10
N=10000
THREATMODEL=Linf

pushd ../..
source activate.sh
popd

srun /usr/bin/nvidia-smi

mkdir -p predictions

python predict.py --model "$MODEL" --dataset "$DATASET" --output "predictions/$DATASET-$MODEL.csv" --n-examples "$N" --threat-model "$THREATMODEL" 
