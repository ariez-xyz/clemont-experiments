#!/bin/bash

# Convert PARAM_PAIRS back to array
IFS=' ' read -r -a param_pairs <<< "$PARAM_PAIRS"

# Get the parameter pair for this task (adjust for 1-based array index)
param_pair="${param_pairs[$((SLURM_ARRAY_TASK_ID-1))]}"

# Split the parameter pair
IFS=',' read -r eps batchsize <<< "$param_pair"

echo epsilon=$eps batchsize=$batchsize

srun python run_on_csv.py "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize "$batchsize" \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--out-path "$results_dir/$batchsize-$eps.json"

