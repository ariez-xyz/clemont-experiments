#!/bin/bash
set -e

echo "Plotting time by ratio percentile..."
python plot_time_by_ratio_percentile.py certifair/baseline_no_maxk/quant_run_20251014T190907.json certifair/fair_no_maxk/quant_run_20251014T155455.json --fill-percentiles 20,80 --bins 20 --no-title --labels 'Naive model,Fair model'

echo "Plotting progressions percentiles..."
python plot_progressions_percentiles.py certifair/fair_kgrow_1p05/

echo "Plotting max ratio combined (certifair)..."
python plot_max_ratio_combined.py certifair/fair_no_maxk/quant_run_20251015T205531.json certifair/baseline_no_maxk/quant_run_20251015T192207.json --output certifair/ --labels 'Fair model,Naive model' --bins 40 --no-title

echo "Plotting max ratio combined (robustbench baseline)..."
python plot_max_ratio_combined.py robustbench_cifar10/baseline/quant_run_20251015T143006.json --split 9999 --output robustbench_cifar10/hist_naive.png --bins 40 --labels 'Real images,Adversarial examples' --no-title

echo "Plotting max ratio combined (robustbench robust)..."
python plot_max_ratio_combined.py robustbench_cifar10/robust/quant_run_20251015T142852.json --split 9999 --output robustbench_cifar10/hist_robust.png --bins 40 --labels 'Real images,Adversarial examples' --no-title

echo "Plotting max ratio combined (eps 0.02)..."
python plot_max_ratio_combined.py certifair/fair_eps_0p020_maxk_none/quant_run_20251015T205339.json --split-epsilon-flagged --no-title --output certifair/hist_flag_vs_nonflag_eps_0.02.png --bins 40

echo "Plotting max ratio combined (eps 0.1)..."
python plot_max_ratio_combined.py certifair/fair_eps_0p100_maxk_none/quant_run_20251015T204847.json --split-epsilon-flagged --no-title --output certifair/hist_flag_vs_nonflag_eps_0.1.png --bins 40

echo "Plotting quant/qual MSE..."
 python plot_epsilon_flag_alignment.py certifair --series "fair=fair*none" --series "base=base*none" --output certifair/fair_vs_base_alignment.png

echo "Plotting time per sample..."
python plot_time_per_sample.py robusttrees_higgs/results/ \
	--run-dir robusttrees_higgs/results/eps_0_01/maxk_128/batch_50000_wt_24-00-00 \
	--run-dir robusttrees_higgs/results/eps_0_01/maxk_256/batch_50000_wt_24-00-00 \
	--run-dir robusttrees_higgs/results/eps_0_01/maxk_512/batch_50000_wt_24-00-00 \
	--run-dir robusttrees_higgs/results/eps_0_01/maxk_1024/batch_50000_wt_72-00-00 \
	--run-dir robusttrees_higgs/results/eps_0_01/maxk_none/batch_50000_wt_72-00-00 \
	--run-dir robusttrees_higgs/results/naive/wt_20-00-00 \
	--labels '128,256,512,1024,none,brute-force' \
	--epsilon-run-dir robusttrees_higgs/results/eps_0_025/maxk_128/batch_50000_wt_24-00-00 \
	--epsilon-labels '$\epsilon$=0.025' \
	--output robusttrees_higgs/combined_time_per_sample.png \
	--no-title \
	--rolling-average 100000
