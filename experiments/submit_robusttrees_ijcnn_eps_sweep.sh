#!/bin/bash

# Main Configuration
export robust_input="../data/RobustTrees/predictions/ijcnn/test_pred.csv"
export unrob_input="../data/RobustTrees/predictions/ijcnn.unrob/test_pred.csv"
export results_base="../results/slurm_robusttrees_ijcnn_eps_sweep"
export work_script="slurm_robusttrees_ijcnn_eps_sweep_work.sh"
export array="1-100"

# Housekeeping
export results_dir="$results_base/results"
export logs_dir="$results_base/logs"
unset SLURM_EXPORT_ENV
mkdir -p "$results_base"
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd

# Submit to queue
sbatch --array=$array --job-name=$work_script --output=$logs_dir/%A-%a.log -c 1 --time=0:05:00 --mem=1G --no-requeue --export=ALL $work_script

