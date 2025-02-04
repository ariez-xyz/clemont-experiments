#!/bin/bash

# parameters
export backend="kdtree"
export metric="infinity"
export input_file="../data/RobustTrees/predictions/higgs/train_pred.csv"
export results_base="../results/performance"
export pred="pred"
export epss=(0.01 0.025 0.05)
export batchsizes=(1000 5000 10000 50000 100000)
export maxtime=$((60*60*22))
export parallelize=2

# setup dirs, venv, etc
export work_script="slurm_performance_work.sh"
export results_dir="$results_base/results/$backend"
export logs_dir="$results_base/logs/$backend"
unset SLURM_EXPORT_ENV
mkdir -p "$results_dir"
mkdir -p "$logs_dir"
pushd ..
source activate.sh
popd

declare -a param_pairs
for eps in "${epss[@]}"; do
    for batchsize in "${batchsizes[@]}"; do
        param_pairs+=("$eps,$batchsize")
    done
done
export PARAM_PAIRS="${param_pairs[*]}"
export NUM_TASKS=${#param_pairs[@]}
export array="1-$NUM_TASKS"

# Submit to queue
sbatch \
	--job-name=$work_script \
	--output="$logs_dir/$backend-%A-%a.log" \
	--array=$array \
	-c 4 \
	--time=24:00:00 \
	--mem=32G \
	--no-requeue \
	--export=ALL \
	$work_script 
