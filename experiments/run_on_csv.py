import pandas as pd
import numpy as np
import argparse
import json
import sys
import time
from datetime import datetime

from aimon.backends.bdd import BDD
from aimon.backends.faiss import BruteForce
from aimon.backends.kdtree import KdTree
from aimon.backends.snn import Snn
from aimon.runner import DataframeRunner

np.set_printoptions(suppress=True)

def log(s):
    timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {s}", flush=True)

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
        if diff == 0:
            line = f"{marker}{df.columns[col][:30].rjust(30)}\t{val_i:.6f}\t{val_j:.6f}\t0"
            rest.append(line)
        elif is_close:
            line = f"{marker}{df.columns[col][:30].rjust(30)}\t{val_i:.6f}\t{val_j:.6f}\t{diff:.6f}"
            rest.append(line)
        else:
            # Only apply ANSI color codes if stdout is a terminal
            if sys.stdout.isatty():
                line = f"\033[91m{marker}{df.columns[col][:30].rjust(30)}\t{val_i:.6f}\t{val_j:.6f}\t{diff:.6f}\033[0m"
            else:
                line = f"{marker}{df.columns[col][:30].rjust(30)}\t{val_i:.6f}\t{val_j:.6f}\t{diff:.6f}"
            differing_attrs.append(line)

    # Order: print differences first
    for line in differing_attrs:
        print(line)
    if not diff_only:
        for line in rest:
            print(line)

def make_argparser():
    parser = argparse.ArgumentParser(description='run monitor on csv format predictions')
    parser.add_argument('csvpath', type=str, nargs='+', help='Path to one or more CSV files')
    parser.add_argument('--eps', type=float, help='epsilon')
    parser.add_argument('--n_bins', '--n-bins', type=int, help='Number of bins')
    parser.add_argument('--n_examples', '--n-examples', type=int, default=None, help='Cap the number of samples to process')
    parser.add_argument('--out_path', '--out-path', type=str, help='Path to save output JSON')
    parser.add_argument('--full_output', '--full-output', action='store_true', help='complete json output (timings, concrete counterexample pairs)')
    parser.add_argument('--verbose', action='store_true', help='verbose output (print differences)')
    parser.add_argument('--randomize_order', '--randomize-order', action='store_true', help='Randomize CSV order')
    parser.add_argument('--backend', type=str, default='bf', choices=['bf', 'bdd', 'kdtree', 'snn'], help='which implementation to use as backend')
    parser.add_argument('--blind_cols', '--blind-cols', type=str, help='comma-separated list of sensitive columns, e.g. "race,sex". allows wildcards like "race=*"')
    parser.add_argument('--pred', type=str, default='pred', help='name of the column holding model predictions')
    parser.add_argument('--metric', type=str, default='infinity', help='metric to use. available choices depend on backend')
    parser.add_argument('--max_time', '--max-time', type=float, default=None, help='maximum number of seconds to run before terminating')
    parser.add_argument('--batchsize', type=int, default=None, help='batchsize (kdtree, snn only)')
    return parser
    
if __name__ == "__main__":
    args = make_argparser().parse_args()
    csvpaths = args.csvpath

    # user must provide exactly 1 of eps or n_bins
    if args.eps and args.n_bins:
        raise ValueError("need one of --eps or --n_bins, not both")
    elif not args.n_bins and not args.eps:
        raise ValueError("need one of --eps or --n_bins")

    # infer the respective missing arg
    if args.n_bins:
        args.eps = 1/args.n_bins
    else:
        args.n_bins = int(1/args.eps)

    dfs = []
    for arg in csvpaths:
        # Handle potential newline-separated paths
        paths = arg.split('\n')
        for path in paths:
            path = path.strip()
            if not path:
                continue
            df = pd.read_csv(path)
            # Ensure consistent column structure across dataframes
            if dfs and df.shape[1] != dfs[0].shape[1]:
                raise ValueError(f"CSV at {path} has incompatible column structure")
            dfs.append(df)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    log(f"loaded data of shape {df.shape}.")

    blind_df = None
    if args.blind_cols: # Remove blind columns
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

    num_columns = df.shape[1]
    low_cardinality_cols = [col for col in df.columns if df[col].nunique() < args.n_bins]
    log(f"low-cardinality columns: {low_cardinality_cols}. assuming categorical (i.e, must be exact match)")

    log(f"metric is {args.metric}...")

    if args.backend == 'bdd':
        log(f"initializing BDD backend...")
        assert args.metric == "infinity", f"BDD: unimplemented metric {args.metric}"
        backend = BDD(
            data_sample=df,
            n_bins=args.n_bins,
            decision_col=args.pred,
            categorical_cols=low_cardinality_cols,
            collect_cex=True
        )

    elif args.backend == 'bf':
        log(f"initializing brute force backend...")
        backend = BruteForce(df, args.pred, args.eps, args.metric.lower())

    elif args.backend == 'kdtree':
        log(f"initializing kd-tree backend...")
        if args.batchsize:
            backend = KdTree(df, args.pred, args.eps, args.metric, batchsize=args.batchsize)
        else:
            backend = KdTree(df, args.pred, args.eps, args.metric)

    elif args.backend == 'snn':
        log(f"initializing snn backend...")
        if args.batchsize:
            backend = Snn(df, args.pred, args.eps, batchsize=args.batchsize)
        else:
            backend = Snn(df, args.pred, args.eps)

    runner = DataframeRunner(backend)

    log(f"starting...")
    monitor_positives = sorted(runner.run(df, args.n_examples))

    if args.verbose:
        for pair in monitor_positives:
            if blind_df is not None:
                pretty_print(blind_df, pair[0], pair[1], args.eps, marker="*")
                pretty_print(df, pair[0], pair[1], args.eps, header=False)
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
            'backend_meta': backend.meta,
            'positives': [(int(x), int(y)) for x, y in monitor_positives],
        } 

        if args.full_output:
            out['timings'] = runner.timings

        with open(args.out_path, 'w') as f:
            json.dump(out, f, indent=2)

