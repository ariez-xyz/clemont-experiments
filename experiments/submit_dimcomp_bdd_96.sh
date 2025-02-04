#!/bin/bash

# parameters
export backend="bdd"
export metric="infinity"
export parallelize="95"
export input_file1="../data/RobustBench/predictions/imagenet-Standard_R50.csv"
export input_file2="../data/RobustBench/predictions/imagenet-Standard_R50-adv.csv"
export results_base="../results/dimcomp"
export pred="pred"
# imagenet max. 3*224*224=150528 cols
export samplecols="8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32767 65536 131072"
export eps=0.0157 # > 4/255
export batchsize=100
export maxtime=$((60*60*12))

# setup dirs, venv, etc
export work_script="slurm_dimcomp_work.sh"
export results_dir="$results_base/results/$backend-96t"
export logs_dir="$results_base/logs/$backend-96t"
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
	-c 96 \
	--time=14:00:00 \
	--mem=256G \
	--no-requeue \
	--export=ALL \
	$work_script 
