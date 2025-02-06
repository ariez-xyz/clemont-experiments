#!/bin/bash

# parameters
export backend="kdtree"
export metric="infinity"
export input_file="../data/RobustTrees/predictions/higgs/train_pred.csv"
export results_base="../results/performance"
export pred="pred"
export epss=(0.01 0.025 0.05)
export dropcols=("23:" "11:")
export maxtime=$((60*60*36))
export parallelize=1

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
    for dropcol in "${dropcols[@]}"; do
        param_pairs+=("$eps,$dropcol")
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
	-c 1 \
	--time=38:00:00 \
	--mem=128G \
	--no-requeue \
	--export=ALL \
	$work_script 
