#!/bin/bash

read -ra samplecols_array <<< "$samplecols"
sc="${samplecols_array[$((SLURM_ARRAY_TASK_ID-1))]}"

echo $array samplecols=$samplecols sc=$sc

srun python run_on_csv.py ../data/RobustBench/predictions/proc_imagenet-Standard_R50-combined-"$sc"d.csv-lowdec.csv \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize "$batchsize" \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--pred 0 \
	--parallelize "$parallelize" \
	--out-path "$results_dir/$sc-d-lowdec.json"

srun python run_on_csv.py ../data/RobustBench/predictions/proc_imagenet-Standard_R50-combined-"$sc"d.csv \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize "$batchsize" \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--parallelize "$parallelize" \
	--out-path "$results_dir/$sc-d.json"
