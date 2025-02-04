#!/bin/bash

# parameters
export name="imagenet-Standard"
export input_files=$(find ../data/RobustBench/predictions/ -name "$name*csv")
# >4/255
export eps=0.0157
export pred="pred"
export results_base="../results/adversarial/"
export work_script="slurm_robustbench_work.sh"

# setup dirs, venv, etc
export results_dir="$results_base/results"
export logs_dir="$results_base/logs"
unset SLURM_EXPORT_ENV
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/$name-%A-%a.log" \
	-c 96 \
	--time=24:00:00 \
	--mem=128G \
	--no-requeue \
	--export=ALL \
	$work_script 
