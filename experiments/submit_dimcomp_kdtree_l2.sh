#!/bin/bash

# parameters
export backend="kdtree"
export metric="l2"
export results_base="../results/dimcomp"
export pred="pred"
# imagenet max. 3*224*224=150528 cols
export samplecols="8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 65536"
export eps=0.0315 # > 8/255
export batchsize=500
export maxtime=$((60*60*22))

# setup dirs, venv, etc
export work_script="slurm_dimcomp_work.sh"
export results_dir="$results_base/results/$metric-$backend"
export logs_dir="$results_base/logs/$metric-$backend"
unset SLURM_EXPORT_ENV
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd
export NUM_TASKS=${#samplecols[@]}
export array="1-14"

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/$backend-%A-%a.log" \
	--array=$array \
	-c 1 \
	--time=24:00:00 \
	--mem=512G \
	--no-requeue \
	--export=ALL \
	$work_script 
