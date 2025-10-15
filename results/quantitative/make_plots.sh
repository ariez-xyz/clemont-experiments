#!/bin/bash
source ../../activate.sh

python plot_time_by_ratio_percentile.py certifair/baseline_no_maxk/quant_run_20251014T190907.json certifair/fair_no_maxk/quant_run_20251014T155455.json --fill-percentiles 20,80 --bins 20 --no-title --labels 'Naive model,Fair model'

python plot_progressions_percentiles.py certifair/fair_kgrow_1p05/

python plot_max_ratio_combined.py certifair/fair_no_maxk/ certifair/baseline_no_maxk/ --output certifair/ --labels 'Fair model,Naive model' --bins 30 --no-title

python plot_max_ratio_combined.py robustbench_cifar10/baseline/quant_run_20251015T143006.json --split 9999 --output robustbench_cifar10/hist_naive.png --bins 40 --labels 'Real images,Adversarial examples' --no-title
python plot_max_ratio_combined.py robustbench_cifar10/robust/quant_run_20251015T142852.json --split 9999 --output robustbench_cifar10/hist_robust.png --bins 40 --labels 'Real images,Adversarial examples' --no-title

python plot_max_ratio_combined.py certifair/fair_eps_0p015_maxk64/quant_run_20251014T190025.json --split-epsilon-flagged --no-title --output certifair/hist_flag_vs_nonflag_eps_0.015.png
python plot_max_ratio_combined.py certifair/fair_eps_0p050_maxk64/quant_run_20251014T190158.json --split-epsilon-flagged --no-title --output certifair/hist_flag_vs_nonflag_eps_0.05.png
python plot_max_ratio_combined.py certifair/fair_eps_0p100_maxk64/quant_run_20251014T190007.json --split-epsilon-flagged --no-title --output certifair/hist_flag_vs_nonflag_eps_0.1.png
