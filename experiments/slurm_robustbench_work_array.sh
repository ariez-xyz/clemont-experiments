#!/bin/bash

input_file=$(echo "$input_files" | sed -n "${SLURM_ARRAY_TASK_ID}p")
input_basename=$(basename $input_file .csv)

echo running on $base_file + $input_file

srun python run_on_csv.py "$base_file" "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--metric "$metric" \
	--out_path "$results_dir/$name-$metric-$eps-$input_basename-bf.json"

srun python run_on_csv.py "$base_file" "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--metric "$metric" \
	--backend kdtree \
	--out_path "$results_dir/$name-$metric-$eps-$input_basename-kd.json"

srun python run_on_csv.py "$base_file" "$input_file" \
	--pred "$pred" \
	--eps "$eps" \
	--metric "$metric" \
	--backend snn \
	--out_path "$results_dir/$name-$metric-$eps-$input_basename-snn.json"
