#!/bin/bash

read eps batchsize <<< "${params[$SLURM_ARRAY_TASK_ID]}"

echo epsilon=$eps batchsize=$batchsize

srun python run_on_csv.py "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize "$batchsize" \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--out-path "$results_dir/$batchsize-$eps.json"

