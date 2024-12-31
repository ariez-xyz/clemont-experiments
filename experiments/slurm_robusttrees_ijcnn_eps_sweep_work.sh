#!/bin/bash

# calculate epsilon based on array job ID
eps=$(python3 -c "import sys; print('{:.3f}'.format($SLURM_ARRAY_TASK_ID * 0.001))")

srun python run_on_csv.py "$robust_input" --eps "$eps" --out_path "$results_dir/rob-$eps.json" \
  && python run_on_csv.py "$unrob_input"  --eps "$eps" --out_path "$results_dir/unrob-$eps.json"

