#!/bin/bash

# get input file for this array job
data_files=$(find ../data/RobustBench/predictions/ -name "*csv")
input=$(echo "$data_files" | sed -n "${SLURM_ARRAY_TASK_ID}p")
basename="${input##*/}"
name="${basename%.*}"

echo job number: "$SLURM_ARRAY_TASK_ID"
echo input file: "$input"
echo model: "$name"

for eps in 0.01 0.02 0.04 0.06 0.08 0.1 0.15 0.25 0.35 0.5; do
	srun python run_on_csv.py "$input" --eps "$eps" --out_path "$results_dir/$name-$eps.json" --full_output
done
