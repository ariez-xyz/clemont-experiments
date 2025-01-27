#!/bin/bash

# parameters
export backend="snn"
export metric="l2"
export input_file1="../data/RobustBench/predictions/imagenet-Standard_R50.csv"
export input_file2="../data/RobustBench/predictions/imagenet-Standard_R50.csv"
export results_base="../results/dimcomp"
export pred="pred"
# imagenet max. 3*224*224=150528 cols
export samplecols="8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 65536 131072"
export eps=0.157 # > 4/255
export batchsize=100
export maxtime=$((60*60*12))

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
export array="1-15"

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/$backend-%A-%a.log" \
	--array=$array \
	-c 4 \
	--time=13:00:00 \
	--mem=32G \
	--no-requeue \
	--export=ALL \
	$work_script 
