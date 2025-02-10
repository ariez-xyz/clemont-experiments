#!/bin/bash

read -ra samplecols_array <<< "$samplecols"
sc="${samplecols_array[$((SLURM_ARRAY_TASK_ID-1))]}"

input_file="../data/RobustBench/predictions/proc_imagenet-Standard_R50-combined-"$sc"d.csv-lowdec.csv"

echo $array samplecols=$samplecols sc=$sc infile=$input_file

srun python run_on_csv.py $input_file \
	--pred "$pred" \
	--eps 0.0157 \
	--batchsize 9999 \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--preload 10 \
	--pred 0 \
	--out-path "$results_dir/$sc-d.json"

