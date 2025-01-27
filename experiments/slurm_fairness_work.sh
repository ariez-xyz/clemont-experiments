#!/bin/bash

# Convert PARAM_PAIRS back to array
IFS=' ' read -r -a param_pairs <<< "$PARAM_PAIRS"

# Get the parameter pair for this task (adjust for 1-based array index)
param_pair="${param_pairs[$((SLURM_ARRAY_TASK_ID-1))]}"

# Split the parameter pair
IFS=',' read -r eps input_file <<< "$param_pair"

input_basename=$(basename $input_file .csv)

echo epsilon=$eps input=$input_file out="$input_basename-$eps.json"

srun python run_on_csv.py "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--metric "$metric" \
	--backend "$backend" \
	--full-output \
	--out-path "$results_dir/$eps$input_basename-eps$eps.json"

