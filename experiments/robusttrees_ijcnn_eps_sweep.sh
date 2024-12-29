#!/bin/fish
mkdir -p ../results/robusttrees_ijcnn_eps_sweep
for eps in (seq 0.001 0.001 0.2)
    set eps_str (printf "%.3f" $eps)
    echo $eps_str
    python run_on_csv.py ../data/RobustTrees/predictions/ijcnn/test_pred.csv --max_n 10000 --eps $eps --out_path ../results/robusttrees_ijcnn_eps_sweep/rob-$eps_str.json
    python run_on_csv.py ../data/RobustTrees/predictions/ijcnn.unrob/test_pred.csv --max_n 10000 --eps $eps --out_path ../results/robusttrees_ijcnn_eps_sweep/unrob-$eps_str.json
end
