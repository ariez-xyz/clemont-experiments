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
#SBATCH --job-name=tian-3dcc
#SBATCH --output=tian-3dcc-%j.log
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

MODEL=Tian2022Deeper_DeiT-B
DATASET=imagenet3dcc
N=5000
THREATMODEL=corruptions_3d

pushd ../..
source activate.sh
popd

srun /usr/bin/nvidia-smi

mkdir -p predictions

for SEVERITY in 1 2 3 4 5; do
    for CORRUPTION in near_focus far_focus fog_3d flash color_quant low_light xy_motion_blur z_motion_blur iso_noise bit_error h265_abr h265_crf; do
        python predict.py --model "$MODEL" \
                         --emb-model small \
                         --dataset "$DATASET" \
                         --output "predictions/$DATASET-$MODEL-$CORRUPTION-$SEVERITY.csv" \
                         --n-examples "$N" \
                         --threat-model "$THREATMODEL" \
                         --corruption3d "$CORRUPTION" \
                         --severity "$SEVERITY"
    done
done
