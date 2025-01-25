#!/bin/bash

# parameters
export name="cifar10c-Standard"
export base_file="../data/RobustBench/predictions/$name/vanilla.csv"
export input_files=$(find "../data/RobustBench/predictions/$name" -name "*csv")
# To find sensible eps: look at avg L1/L2/Linf/... distance between the embeddings an image and its corrupted version, vs. avg distance between a random pair of embeddings
export eps=25
export metric="L2"
export pred="pred"
export results_base="../results/semantic/"
export work_script="slurm_robustbench_work_array.sh"
export verbose="false"

# setup dirs, venv, etc
export results_dir="$results_base/results/$name"
export logs_dir="$results_base/logs/$name"
unset SLURM_EXPORT_ENV
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd
# Count data files for array job
export array="1-$(echo "$input_files" | wc -w)"
echo $input_files
echo $array

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/$name-%A-%a.log" \
	--array=$array \
	-c 1 \
	--time=0:15:00 \
	--mem=8G \
	--no-requeue \
	--export=ALL \
	$work_script 
