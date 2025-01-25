#!/bin/bash

# parameters
export name="cifar100-Wang"
export input_files=$(find ../data/RobustBench/predictions/ -name "$name*csv")
# >8/255
export eps=0.031373
export pred="pred"
export results_base="../results/adversarial/"
export work_script="slurm_robustbench_work.sh"
export verbose="false"

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
	-c 1 \
	--time=0:30:00 \
	--mem=4G \
	--no-requeue \
	--export=ALL \
	$work_script 
