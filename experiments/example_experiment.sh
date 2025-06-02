#!/bin/bash

# parameters
export name="cifar10-Standard"
export input_files=$(find ../data/RobustBench/predictions/ -name "$name*csv")
# >8/255
export eps=0.031373
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

python run_on_csv.py "$input_files" \
	--pred "$pred" \
	--eps "$eps" \
	--backend bf \
	--out_path "$results_dir/$name-$eps-bf.json" \
	--metric infinity \
	--full_output

