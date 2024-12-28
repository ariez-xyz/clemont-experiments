import pandas as pd
import numpy as np
import argparse

from aimon.backends.bdd import BDD
from aimon.backends.faiss import BruteForce
from aimon.runner import Runner

np.set_printoptions(suppress=True)

def pprint_pair(df, i, j, eps):
    print(f'\nrow {i}\trow {j}\tdiff\t<{eps}?')
    for col in range(len(df.columns)):
        val_i = df.iloc[i, col]
        val_j = df.iloc[j, col]
        diff = abs(val_i - val_j)
        is_close = diff < eps
        print(f"{val_i:.4f}\t{val_j:.4f}\t{diff:.4f}\t{is_close}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run monitor on csv format predictions')
    parser.add_argument('csvpath', type=str, help='Path to the CSV file')
    parser.add_argument('NBINS', type=int, help='Number of bins')
    parser.add_argument('--full_output', action='store_true', help='Flag to print full output')
    
    args = parser.parse_args()
    csvpath = args.csvpath
    NBINS = args.NBINS

    df = pd.read_csv(csvpath)
    low_cardinality_cols = [col for col in df.columns if df[col].nunique() < NBINS]
    print(f"assuming {low_cardinality_cols} for categorical attributes (must be exact match)")
    num_columns = df.shape[1]

    bdd = BDD(
        data_sample=df,
        n_bins=NBINS,
        decision_col='pred',
        categorical_cols=low_cardinality_cols,
        collect_cex=True
    )

    bf = BruteForce(df, 'pred', 1/NBINS)

    #runner = Runner(bdd)
    runner = Runner(bf)

    monitor_positives = sorted(runner.run(df))

    print(f'found {runner.n_true_positives} unfair pairs')

    if args.full_output:
        print(monitor_positives)
        for pair in monitor_positives:
            pprint_pair(df, pair[0], pair[1], 1/NBINS)

