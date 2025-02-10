#!/bin/bash

# Convert PARAM_PAIRS back to array
IFS=' ' read -r -a param_pairs <<< "$PARAM_PAIRS"

# Get the parameter pair for this task (adjust for 1-based array index)
param_pair="${param_pairs[$((SLURM_ARRAY_TASK_ID-1))]}"

# Split the parameter pair
IFS=',' read -r eps dropcols <<< "$param_pair"

echo epsilon=$eps dropcols=$dropcols

srun python run_on_csv.py "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize 10000 \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--parallelize "$parallelize" \
	--blind-cols $dropcols \
	--randomize-order \
	--out-path "$results_dir/$metric-$eps-$parallelize-$dropcols-run3.json"

