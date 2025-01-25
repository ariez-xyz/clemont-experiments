#!/bin/bash

if [ "$verbose" = "true" ]; then
	srun python run_on_csv.py "$input_files" \
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
