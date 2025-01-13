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
#SBATCH --job-name=wang-cifar100
#SBATCH --output=wang-cifar100-%j.log
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

pushd ../..
source activate.sh
popd

srun /usr/bin/nvidia-smi

srun ./infer.sh Wang2023Better_WRN-70-16 cifar100 10000
