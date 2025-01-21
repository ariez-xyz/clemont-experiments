#!/bin/bash

# output dirs
export results_base="../results/robustbench/"
export work_script="slurm_robustbench_work.sh"

## find and count all prediction csv's in data directory
#export data_files=$(for dir in $(find ../data/ -maxdepth 1 -mindepth 1 -type d); do find $dir/predictions/ -name *csv; done)

# RobustBench only
export data_files=$(find ../data/RobustBench/predictions_dinobase/ -name *csv)
#                  Count data files
export array="1-$(echo "$data_files" | wc -w)"

# setup dirs, venv, etc
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
sbatch --array=$array --job-name=$work_script --output=$logs_dir/%A-%a.log -c 4 --time=0:30:00 --mem=16G --no-requeue --export=ALL $work_script

