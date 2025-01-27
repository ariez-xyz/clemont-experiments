#!/bin/bash

for dataset in adult compas german; do
    for type in base global local; do
        for period in P1 P2; do
            # Combine files while keeping only the first header
            head -n 1 "$dataset-$type-$period-train.csv" > "$dataset-$type-$period-combined.csv"
            tail -n +2 "$dataset-$type-$period-train.csv" >> "$dataset-$type-$period-combined.csv"
            tail -n +2 "$dataset-$type-$period-test.csv" >> "$dataset-$type-$period-combined.csv"
        done
    done
done
