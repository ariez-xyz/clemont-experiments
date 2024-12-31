#!/bin/bash
pushd ..
source activate.sh
popd

# Calculate epsilon value based on array task ID
eps=$(echo "scale=3; $SLURM_ARRAY_TASK_ID * 0.001" | bc)
eps_str=$(printf "%.3f" $eps)
echo "$eps_str"

python run_on_csv.py "$robust_input" --max_n 10000 --eps "$eps" --out_path "$results_dir/rob-$eps_str.json"
python run_on_csv.py "$unrob_input" --max_n 10000 --eps "$eps" --out_path "$results_dir/unrob-$eps_str.json"

