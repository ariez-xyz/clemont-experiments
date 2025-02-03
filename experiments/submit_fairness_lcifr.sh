#!/bin/bash

# parameters
export metric="infinity"
export input_files=($(find "../data/lcifr/predictions/" -name "pred*csv"))
export epss=(0.0025 0.005 0.01 0.02 0.04 0.08 0.16 0.32)

export results_base="../results/fairness/"
export pred="Prediction"

# setup dirs, venv, etc
export work_script="slurm_fairness_work.sh"
export results_dir="$results_base/results/lcifr/"
export logs_dir="$results_base/logs/lcifr/"
unset SLURM_EXPORT_ENV
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd

declare -a param_pairs
for eps in "${epss[@]}"; do
    for input_file in "${input_files[@]}"; do
        param_pairs+=("$eps,$input_file")
    done
done
export PARAM_PAIRS="${param_pairs[*]}"
export NUM_TASKS=${#param_pairs[@]}
export array="1-$NUM_TASKS"

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/%a-%A.log" \
	--array=$array \
	-c 16 \
	--time=01:00:00 \
	--mem=1G \
	--no-requeue \
	--export=ALL \
	$work_script 
