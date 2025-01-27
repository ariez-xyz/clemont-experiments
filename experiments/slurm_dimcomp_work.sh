#!/bin/bash

read -ra samplecols_array <<< "$samplecols"
sc="${samplecols_array[$((SLURM_ARRAY_TASK_ID-1))]}"

echo $array samplecols=$samplecols sc=$sc

srun python run_on_csv.py "$input_file1" "$input_file2" \
	--pred "$pred" \
	--eps "$eps" \
	--batchsize "$batchsize" \
	--metric "$metric" \
	--max-time "$maxtime" \
	--backend "$backend" \
	--full-output \
	--sample-cols "$sc" \
	--blind-cols label \
	--out-path "$results_dir/$sc-d.json"

