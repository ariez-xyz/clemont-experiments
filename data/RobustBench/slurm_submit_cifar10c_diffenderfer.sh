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
#SBATCH --job-name=diffenderfer-cifar10c
#SBATCH --output=diffenderfer-cifar10c-%j.log
#
# 8 cores
#SBATCH -c 8
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#
# Do not requeue the job in the case it fails.
#SBATCH --no-requeue
# Export the local environment to the compute nodes
#SBATCH --export=ALL
#

MODEL=Diffenderfer2021Winning_LRR_CARD_Deck
DATASET=cifar10c
N=10000
THREATMODEL=corruptions

pushd ../..
source activate.sh
popd

srun /usr/bin/nvidia-smi

mkdir -p predictions

for SEVERITY in 1 2 3 4 5; do
    for CORRUPTION in shot_noise motion_blur snow pixelate gaussian_noise defocus_blur brightness fog zoom_blur frost glass_blur impulse_noise contrast jpeg_compression elastic_transform; do
        python predict.py --model "$MODEL" \
                         --emb-model base \
                         --dataset "$DATASET" \
                         --output "predictions/$DATASET-$MODEL-$CORRUPTION-$SEVERITY.csv" \
                         --n-examples "$N" \
                         --threat-model "$THREATMODEL" \
                         --corruption "$CORRUPTION" \
                         --severity "$SEVERITY"
    done
done
