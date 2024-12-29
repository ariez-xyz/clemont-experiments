import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime

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
    parser.add_argument('--eps', type=float, help='epsilon')
    parser.add_argument('--n_bins', type=int, help='Number of bins')
    parser.add_argument('--max_n', type=int, help='Cap the number of samples to process', default=-1)
    parser.add_argument('--out_path', type=str, help='Path to save output JSON')
    parser.add_argument('--full_output', action='store_true', help='Flag to print full output')
    parser.add_argument('--randomize_order', action='store_true', help='Randomize CSV order')
    
    args = parser.parse_args()
    csvpath = args.csvpath

    # user must provide exactly 1 of eps or n_bins
    if args.eps and args.n_bins:
        print("need one of --eps or --n_bins, not both")
        exit(1)
    elif not args.n_bins and not args.eps:
        print("need one of --eps or --n_bins")
        exit(1)

    # infer the respective missing arg
    if args.n_bins:
        args.eps = 1/args.n_bins
    else:
        args.n_bins = int(1/args.eps)

    df = pd.read_csv(csvpath) 
    if args.randomize_order:
        df = df.sample(frac=1).reset_index(drop=True)  # Randomize the dataframe rows
    low_cardinality_cols = [col for col in df.columns if df[col].nunique() < args.n_bins]
    print(f"assuming {low_cardinality_cols} for categorical attributes (must be exact match)")
    num_columns = df.shape[1]

    bdd = BDD(
        data_sample=df,
        n_bins=args.n_bins,
        decision_col='pred',
        categorical_cols=low_cardinality_cols,
        collect_cex=True
    )

    bf = BruteForce(df, 'pred', args.eps)

    #runner = Runner(bdd)
    runner = Runner(bf)

    monitor_positives = sorted(runner.run(df, args.max_n))

    print(f'found {runner.n_true_positives} unfair pairs')

    if args.out_path:
        out = {
            'n_true_positives': runner.n_true_positives,
            'n_positives': runner.n_positives,
            'total_time': runner.total_time,
            'date': datetime.now().isoformat(),
            'backend': runner.get_backend_name(),
            'n_bins': args.n_bins,
            'eps': args.eps,
            'args': vars(args),
        } 

        if args.full_output:
            out['positives'] = [(int(x), int(y)) for x, y in monitor_positives]
            out['timings'] = runner.timings

        with open(args.out_path, 'w') as f:
            json.dump(out, f, indent=2)

    elif args.full_output:
        print(monitor_positives)
        for pair in monitor_positives:
            pprint_pair(df, pair[0], pair[1], args.eps)

