#!/bin/bash

# parameters
export backend="kdtree"
export metric="infinity"
export input_file="../data/RobustTrees/predictions/ijcnn/test_pred.csv"
export results_base="../results/performance/ijcnn"
export pred="pred"
export epss=(0.005 0.01 0.02 0.04 0.08)
export batchsizes=(500 1000 2000 4000 8000)
export maxtime=$((60*60*2))

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
	-c 1 \
	--time=0:15:00 \
	--mem=8G \
	--no-requeue \
	--export=ALL \
	$work_script 
