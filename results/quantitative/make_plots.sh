#!/bin/bash
source ../../activate.sh
python plot_time_by_ratio_percentile.py certifair/baseline_no_maxk/quant_run_20251014T190907.json certifair/fair_no_maxk/quant_run_20251014T155455.json --fill-percentiles 20,80 --bins
python plot_progressions_percentiles.py certifair/fair_kgrow_1p05/
