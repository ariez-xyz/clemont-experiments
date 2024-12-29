#!/bin/fish

set robust_input "../data/RobustTrees/predictions/ijcnn/test_pred.csv"
set unrob_input "../data/RobustTrees/predictions/ijcnn.unrob/test_pred.csv"
set results_base "../results/robusttrees_ijcnn_eps_sweep"
set results_dir "$results_base/results"

mkdir -p $results_base
mkdir -p $results_dir

for eps in (seq 0.001 0.001 0.2)
    set eps_str (printf "%.3f" $eps)
    echo $eps_str
    python run_on_csv.py $robust_input --max_n 10000 --eps $eps --out_path $results_dir/rob-$eps_str.json
    python run_on_csv.py $unrob_input --max_n 10000 --eps $eps --out_path $results_dir/unrob-$eps_str.json
end
