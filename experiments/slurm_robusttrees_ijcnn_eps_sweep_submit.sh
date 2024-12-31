#!/bin/bash

# Main Configuration
robust_input="../data/RobustTrees/predictions/ijcnn/test_pred.csv"
unrob_input="../data/RobustTrees/predictions/ijcnn.unrob/test_pred.csv"
results_base="../results/robusttrees_ijcnn_eps_sweep_slurm"
work_script="slurm_robusttrees_ijcnn_eps_sweep_work.sh"
array="1,3"

# Housekeeping
results_dir="$results_base/results"
logs_dir="$results_base/logs"
unset SLURM_EXPORT_ENV
mkdir -p "$results_base"
mkdir -p "$results_dir"
mkdir -p "$logs_dir"

# Submit to queue
sbatch --array=$array --job-name=$work_script --output=$logs_dir/%A-%a.log -c 1 --time=0:01:00 --mem=1G --no-requeue --export=NONE "srun $work_script"

