#!/bin/bash

for dataset in adult compas german; do
    for type in base global local; do
        for p in P1 P2; do
            # Combine files while keeping only the first header
            head -n 1 "$dataset-$type-$p-train.csv" > "$dataset-$type-$p-combined.csv"
            tail -n +2 "$dataset-$type-$p-train.csv" >> "$dataset-$type-$p-combined.csv"
            tail -n +2 "$dataset-$type-$p-test.csv" >> "$dataset-$type-$p-combined.csv"
        done
    done
done
