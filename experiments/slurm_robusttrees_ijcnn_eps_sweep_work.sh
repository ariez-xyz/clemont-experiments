#!/bin/bash

# calculate epsilon based on array job ID
eps=$(python3 -c "import sys; print('{:.3f}'.format($SLURM_ARRAY_TASK_ID * 0.001))")

srun python run_on_csv.py "$robust_input" --max_n 10000 --eps "$eps" --out_path "$results_dir/rob-$eps.json" --randomize_order \
  && python run_on_csv.py "$unrob_input"  --max_n 10000 --eps "$eps" --out_path "$results_dir/unrob-$eps.json" --randomize_order

