#!/bin/bash
#SBATCH --job-name=robusttrees_ijcnn_eps_sweep
#SBATCH --output=robusttrees_ijcnn_eps_sweep-%A-%a.log
#SBATCH -c 1
#SBATCH --time=0:01:00
#SBATCH --mem=1G
# Do not requeue if job fails:
#SBATCH --no-requeue
# Do not export local env to job: (?)
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

echo $(pwd)
srun ??

robust_input="../data/RobustTrees/predictions/ijcnn/test_pred.csv"
unrob_input="../data/RobustTrees/predictions/ijcnn.unrob/test_pred.csv"
results_base="../results/robusttrees_ijcnn_eps_sweep_slurm"
results_dir="$results_base/results"

mkdir -p "$results_base"
mkdir -p "$results_dir"

# Calculate epsilon value based on array task ID
eps=$(echo "scale=3; $SLURM_ARRAY_TASK_ID * 0.001" | bc)
eps_str=$(printf "%.3f" $eps)
echo "$eps_str"

python run_on_csv.py "$robust_input" --max_n 10000 --eps "$eps" --out_path "$results_dir/rob-$eps_str.json"
python run_on_csv.py "$unrob_input" --max_n 10000 --eps "$eps" --out_path "$results_dir/unrob-$eps_str.json"

