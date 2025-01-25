#!/bin/bash

srun python run_on_csv.py "$data_files" --eps "$eps" --out_path "$results_dir/$name-$eps.json" --full_output --verbose

