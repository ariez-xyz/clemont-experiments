import pandas as pd
import numpy as np
import argparse
import json
import time
from datetime import datetime

from aimon.backends.bdd import BDD
from aimon.backends.faiss import BruteForce
from aimon.runner import Runner

np.set_printoptions(suppress=True)

def log(s):
    timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {s}", flush=True)

def fatal(s):
    log("fatal: " + s)
    exit(1)

def pretty_print(df, i, j, eps, header=True, diff_only=False, marker=" "):
    if header:
        print(f'\n {"column".rjust(30)}\trow {i}\trow {j}\tdelta')

    differing_attrs = []
    rest = []
    for col in range(len(df.columns)):
        val_i = df.iloc[i, col]
        val_j = df.iloc[j, col]
        diff = abs(val_i - val_j)
        is_close = diff < eps
        if is_close:
            line = f"{marker}{df.columns[col][:30].rjust(30)}\t{val_i:.6f}\t{val_j:.6f}\t{diff:.6f}"
            rest.append(line)
        else:
            line = f"\033[91m{marker}{df.columns[col][:30].rjust(30)}\t{val_i:.6f}\t{val_j:.6f}\t{diff:.6f}\033[0m"
            differing_attrs.append(line)

    for line in differing_attrs:
        print(line)
    if not diff_only:
        for line in rest:
            print(line)

def make_argparser():
    parser = argparse.ArgumentParser(description='run monitor on csv format predictions')
    parser.add_argument('csvpath', type=str, help='Path to the CSV file')
    parser.add_argument('--eps', type=float, help='epsilon')
    parser.add_argument('--n_bins', '--n-bins', type=int, help='Number of bins')
    parser.add_argument('--n_examples', '--n-examples', type=int, help='Cap the number of samples to process', default=-1)
    parser.add_argument('--out_path', '--out-path', type=str, help='Path to save output JSON')
    parser.add_argument('--full_output', '--full-output', action='store_true', help='verbose output (timings, concrete counterexample pairs)')
    parser.add_argument('--randomize_order', '--randomize-order', action='store_true', help='Randomize CSV order')
    parser.add_argument('--backend', type=str, default='bf', choices=['bf', 'bdd'], help='which implementation to use as backend')
    parser.add_argument('--blind_cols', '--blind-cols', type=str, help='comma-separated list of sensitive columns, e.g. "race,sex". allows wildcards like "race=*"')
    parser.add_argument('--pred', type=str, default='pred', help='name of the column holding model predictions')
    return parser
    
if __name__ == "__main__":
    args = make_argparser().parse_args()
    csvpath = args.csvpath

    # user must provide exactly 1 of eps or n_bins
    if args.eps and args.n_bins:
        fatal("need one of --eps or --n_bins, not both")
    elif not args.n_bins and not args.eps:
        fatal("need one of --eps or --n_bins")

    # infer the respective missing arg
    if args.n_bins:
        args.eps = 1/args.n_bins
    else:
        args.n_bins = int(1/args.eps)

    log(f"loading {csvpath}...")
    df = pd.read_csv(csvpath) 

    log(f"loaded data of shape {df.shape}.")

    if args.blind_cols:
        cols_to_drop = []
        for expr in args.blind_cols.split(","):
            if expr[-1] == "*": # wildcard
                for col in df.columns:
                    if col.startswith(expr[:-1]):
                        cols_to_drop.append(col)
            else: # literal
                cols_to_drop.append(expr)
        # Separate into dropped and remaining columns
        blind_df = df[cols_to_drop].copy()
        df.drop(columns=cols_to_drop, inplace=True)
        log(f"dropped sensitive columns {cols_to_drop} (new shape is {df.shape})")

    if args.randomize_order:
        df = df.sample(frac=1).reset_index(drop=True)  # Randomize the dataframe rows
        log(f"randomized order")

    low_cardinality_cols = [col for col in df.columns if df[col].nunique() < args.n_bins]
    log(f"low-cardinality columns: {low_cardinality_cols}. assuming categorical (i.e, must be exact match)")

    num_columns = df.shape[1]

    if args.backend == 'bdd':
        log(f"initializing BDD backend...")
        backend = BDD(
            data_sample=df,
            n_bins=args.n_bins,
            decision_col=args.pred,
            categorical_cols=low_cardinality_cols,
            collect_cex=True
        )
    elif args.backend == 'bf':
        log(f"initializing brute force backend...")
        backend = BruteForce(df, args.pred, args.eps)

    runner = Runner(backend)

    log(f"starting...")
    monitor_positives = sorted(runner.run(df, args.n_examples))

    if args.full_output:
        for pair in monitor_positives:
            if blind_df is not None:
                pretty_print(df, pair[0], pair[1], args.eps)
                pretty_print(blind_df, pair[0], pair[1], args.eps, marker="*", header=False)
            else:
                pretty_print(df, pair[0], pair[1], args.eps)

    log(f'found {runner.n_true_positives} unfair pairs')

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


