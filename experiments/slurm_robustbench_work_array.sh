#!/bin/bash

input_file=$(echo "$input_files" | sed -n "${SLURM_ARRAY_TASK_ID}p")

echo running on $base_file + $input_file

if [ "$verbose" = "true" ]; then
	srun python run_on_csv.py "$base_file" "$input_file" \
		--pred "$pred" \
		--eps "$eps" \
		--out_path "$results_dir/$name-$eps.json" \
		--full_output
		--verbose
else
	srun python run_on_csv.py "$input_files" \
		--pred "$pred" \
		--eps "$eps" \
		--out_path "$results_dir/$name-$eps.json" \
		--full_output
fi
