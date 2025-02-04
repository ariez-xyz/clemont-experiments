#!/bin/bash

srun python run_on_csv.py "$input_files" \
	--pred "$pred" \
	--eps "$eps" \
	--backend bf \
	--out_path "$results_dir/$name-$eps-bf.json" \
	--metric infinity \
	--full_output

srun python run_on_csv.py "$input_files" \
	--pred "$pred" \
	--eps "$eps" \
	--backend kdtree \
	--out_path "$results_dir/$name-$eps-kdtree.json" \
	--metric infinity \
	--full_output

srun python run_on_csv.py "$input_files" \
	--pred "$pred" \
	--eps "$eps" \
	--backend kdtree \
	--parallelize 15 \
	--out_path "$results_dir/$name-$eps-kdtree-15.json" \
	--metric infinity \
	--full_output

srun python run_on_csv.py "$input_files" \
	--pred "$pred" \
	--eps "$eps" \
	--backend kdtree \
	--parallelize 95 \
	--out_path "$results_dir/$name-$eps-kdtree-96.json" \
	--metric infinity \
	--full_output

srun python run_on_csv.py "$input_files" \
	--pred "$pred" \
	--eps "$eps" \
	--backend bdd \
	--parallelize 95 \
	--out_path "$results_dir/$name-$eps-bdd-96.json" \
	--metric infinity \
	--full_output
