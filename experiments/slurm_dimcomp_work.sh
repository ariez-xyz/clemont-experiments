#!/bin/bash

read -ra samplecols_array <<< "$samplecols"
sc="${samplecols_array[$((SLURM_ARRAY_TASK_ID-1))]}"

echo $array samplecols=$samplecols sc=$sc

srun python run_on_csv.py ../data/RobustBench/predictions/imagenet-Standard_R50.csv \
	../data/RobustBench/predictions/imagenet-Standard_R50-adv.csv \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize "$batchsize" \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--parallelize "$parallelize" \
	--sample-cols "$sc" \
	--blind-cols label \
	--out-path "$results_dir/$sc-d.json"

