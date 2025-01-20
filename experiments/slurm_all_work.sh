#!/bin/bash

# get input file for this array job
input=$("$data_files" | tr ' ' '\n' | sed -n "${i}p")

srun python run_on_csv.py "$input" --eps "$eps" --out_path "$results_dir/rob-$eps.json"

