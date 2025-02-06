#!/bin/bash

# parameters
export backend="bdd"
export metric="infinity"
export results_base="../results/dimcomp"
export pred="pred"
# imagenet max. 3*224*224=150528 cols
export samplecols="8 16 32 64 128 256"
export eps=0.0315 # > 8/255
export batchsize=0
export maxtime=$((60*60*38))

# setup dirs, venv, etc
export work_script="slurm_dimcomp_work.sh"
export results_dir="$results_base/results/$backend"
export logs_dir="$results_base/logs/$backend"
unset SLURM_EXPORT_ENV
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd
export NUM_TASKS=${#samplecols[@]}
export array="1-6"

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/$backend-%A-%a.log" \
	--array=$array \
	-c 1 \
	--time=36:00:00 \
	--mem=512G \
	--no-requeue \
	--export=ALL \
	$work_script 
