import pandas as pd
import numpy as np
import argparse
import json
import sys
import time
import resource
from datetime import datetime
from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

from aimon.backends.bdd import BDD, BAD_CHARS
from aimon.backends.faiss import BruteForce
from aimon.backends.kdtree import KdTree
from aimon.backends.snn import Snn
from aimon.runner import DataframeRunner

np.set_printoptions(suppress=True)

def debug(s):
    if False:
        timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print(f"[DEBUG {timestamp}] {s}", flush=True)

def log(s):
    timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {s}", flush=True)

def pretty_print(df, i, j, eps, header=True, diff_only=False, marker=" ", metric="infinity"):
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

def setup_backend(args, df):
    global start_time
    if args.backend == 'bdd':
        assert args.metric == "infinity", f"BDD: unimplemented metric {args.metric}"
        low_cardinality_cols = [col for col in df.columns if df[col].nunique() < args.n_bins]
        log(f"low-cardinality columns: {low_cardinality_cols}. assuming categorical (i.e, must be exact match)")
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
        backend = BruteForce(df, args.pred, args.eps, args.metric.lower(), args.faiss_omp_threads)

    elif args.backend == 'kdtree':
        log(f"initializing kd-tree backend...")
        if args.batchsize:
            backend = KdTree(df, args.pred, args.eps, args.metric.lower(), batchsize=args.batchsize, bf_threads=args.st_threads)
        else:
            backend = KdTree(df, args.pred, args.eps, args.metric.lower(), bf_threads=args.st_threads)

    elif args.backend == 'snn':
        log(f"initializing snn backend...")
        assert args.metric.lower() == "l2", f"SNN: unimplemented metric {args.metric}"
        if args.batchsize:
            backend = Snn(df, args.pred, args.eps, batchsize=args.batchsize, bf_threads=args.st_threads)
        else:
            backend = Snn(df, args.pred, args.eps, bf_threads=args.st_threads)
    else:
        raise ValueError(f"unknown backend {args.backend}")

    if args.preload:
        shape = backend.preload(df, args.pred, repeat=args.preload)
        log(f"EXPERIMENTAL: preloaded data of shape {shape}")
        start_time = time.time()

    return DataframeRunner(backend)

def partition(df, n_splits, pred):
    """
    Partition df into n_splits many sub-df's by splitting columns
    Every partition contains the 'pred' column as well.
    """

    pred_idx = df.columns.get_loc(pred)  # Index of the pred column

    other_idxs = np.array([i for i in range(df.shape[1]) if i != pred_idx])  # All indices except pred
    permuted_other = np.random.default_rng(42).permutation(other_idxs)
    
    cols_per_partition = np.array_split(permuted_other, n_splits)
    
    partitions = []
    for subset in cols_per_partition:
        partition_indices = np.append(subset, pred_idx) # Add the pred column to each partition
        partitions.append(df.iloc[:, partition_indices])

    log(f"partitioned {len(df.columns)} columns: {list(df.columns)}")
    return partitions

def process_partition(args, df, idx, result_queue, msg_batch=100):
    log(f"\tworker {idx}: {len(df.columns)} columns {list(df.columns)}")

    def send(msg_type, payload): 
        result_queue.put((msg_type, idx, payload))
        if msg_type == "iter":
            debug(f"\t worker {idx}: send ({cexs_buf})")

    runner = setup_backend(args, df)
    cex_sizes = []
    cexs_buf = []
    last_updated = 0

    send("iter", (0, [])) # Indicate worker is ready

    for step, iter_cexs in enumerate(runner.run(df, args.n_examples, max_time=args.max_time)):
        cexs_buf.extend(iter_cexs)
        if len(cexs_buf) >= msg_batch or step - last_updated > 1000:
            last_updated = step
            send("iter", (step, cexs_buf))
            cexs_buf = []

        cex_sizes.append(len(iter_cexs))

    # send remaining updates
    send("iter", (len(cex_sizes), cexs_buf))
    send("metrics", collect_metrics(runner, args))

    log(f"\tworker {idx}: finished, sent {sum(cex_sizes)/len(cex_sizes)} cexs average")
    debug(f"{cex_sizes}")

def collect_metrics(runner, args):
    metrics = {
        'n_true_positives': runner.n_true_positives,
        'n_positives': runner.n_positives,
        'n_processed': len(runner.timings),
        'n_flagged': runner.n_flagged,
        'perc_flagged': runner.n_flagged / len(runner.timings),
        'data_shape': runner.data_shape,
        'total_time': runner.total_time,
        'avg_time': runner.total_time / len(runner.timings),
        'date': datetime.now().isoformat(),
        'peak_mem': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'backend': runner.get_backend_name(),
        'n_bins': args.n_bins,
        'eps': args.eps,
        'args': vars(args),
        'backend_meta': runner.backend.meta,
    } 

    if args.full_output:
        metrics['timings'] = [round(t, 6) for t in runner.timings]
        metrics['mem'] = runner.mem
        metrics['radius_query_ks'] = getattr(runner.backend, 'radius_query_ks', [])

    return metrics

def make_argparser():
    parser = argparse.ArgumentParser(description='run monitor on csv format predictions')
    parser.add_argument('csvpath', type=str, nargs='+', help='Path to one or more CSV files')
    parser.add_argument('--eps', type=float, help='epsilon')
    parser.add_argument('--n_bins', '--n-bins', type=int, help='Number of bins')
    parser.add_argument('--n_examples', '--n-examples', type=int, default=None, help='Cap the number of samples to process')
    parser.add_argument('--out_path', '--out-path', type=str, help='Path to save output JSON')
    parser.add_argument('--full_output', '--full-output', action='store_true', help='complete json output (timings)')
    parser.add_argument('--concise_output', '--concise-output', action='store_true', help='shortened json output (omit positives)')
    parser.add_argument('--verbose', action='store_true', help='verbose output (print differences)')
    parser.add_argument('--randomize_order', '--randomize-order', action='store_true', help='Randomize CSV order')
    parser.add_argument('--backend', type=str, default='bf', choices=['bf', 'bdd', 'kdtree', 'snn'], help='which implementation to use as backend')
    parser.add_argument('--parallelize', type=int, default=1, help='split data to multiple processes (Linf only).')
    parser.add_argument('--blind_cols', '--blind-cols', type=str, help='comma-separated list of column names for the monitor to ignore, e.g. "race,sex". allows * wildcard, e.g. "race=*" to drop all columns starting with "race=". allows slicing, e.g. "12:" to drop all columns that come after column 12')
    parser.add_argument('--sample_cols', '--sample-cols', type=int, default=None, help='integer number of columns to randomly sample from the data (will discard the other columns)')
    parser.add_argument('--pred', type=str, default='pred', help='name of the column holding model predictions')
    parser.add_argument('--metric', type=str, default='infinity', help='metric to use. available choices depend on backend')
    parser.add_argument('--max_time', '--max-time', type=float, default=None, help='maximum number of seconds to run before terminating')
    parser.add_argument('--batchsize', type=int, default=None, help='batchsize (kdtree, snn only)')
    parser.add_argument('--st_threads', '--st-threads', type=int, default=1, help='number of threads to use for the brute force short-term memory (kdtree, snn backends only). defaults to 1')
    parser.add_argument('--faiss_omp_threads', '--faiss-omp-threads', type=int, default=0, help='number of faiss threads to use (bf backend only). defaults to max available')
    parser.add_argument('--diff', type=str, default=None, help='path to JSON output file to diff the positives against')
    parser.add_argument('--pairwise_diff', '--pairwise-diff', type=str, default=None, help='path to JSON output file to diff the positives against; pairwise output')
    parser.add_argument('--preload', type=int, default=None, help='EXPERIMENTAL. Breaks batching, timing, and more. Augment the CSVs with noise and preload them into backend this many times')
    parser.add_argument('--quantize_pred', '--quantize-pred', type=int, default=None, help='EXPERIMENTAL. Reduce unique values in prediction column to specified number (for benchmarking).')
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
        if args.backend != "bdd":
            log(f"warning: converted n_bins={args.n_bins} to eps={args.eps}, assuming data is in [0,1] across all dimensions")
    else:
        args.n_bins = int(1/args.eps)
        if args.backend == "bdd":
            log(f"warning: converted eps={args.eps} to {args.n_bins} bins, assuming data is in [0,1] across all dimensions")

    ######################
    # DATA PREPROCESSING #
    ######################

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
            elif expr[-1] == ":": # slice
                drop = False
                for col in df.columns:
                    if drop:
                        cols_to_drop.append(col)
                    elif col.startswith(expr[:-1]):
                        drop = True
            else: # literal
                cols_to_drop.append(expr)
        # Separate into dropped and remaining columns
        blind_df = df[cols_to_drop].copy()
        df.drop(columns=cols_to_drop, inplace=True)
        log(f"dropped columns {cols_to_drop} (new shape is {df.shape})")

    if args.sample_cols is not None:
        # Ensure prediction column is included in sample
        non_pred_cols = df.columns.drop(args.pred)
        sampled_columns = non_pred_cols.to_series().sample(n=args.sample_cols, random_state=0)
        keep = pd.concat([sampled_columns, pd.Series([args.pred])])
        df = df[keep]
        log(f"keeping columns: {list(keep)} (new shape is {df.shape})")

    if args.randomize_order:
        df = df.sample(frac=1).reset_index(drop=True)  # Randomize the dataframe rows
        log(f"randomized order")

    if args.backend == 'bdd': # Rename columns
        old_cols = df.columns.tolist()
        for old_char, new_char in BAD_CHARS.items():
            df.columns = df.columns.str.replace(old_char, new_char)
        new_cols = df.columns.tolist()
        changed = [(old, new) for old, new in zip(old_cols, new_cols) if old != new]
        if changed:
            log("Renamed columns for BDD compatibility:")
            for old, new in changed:
                log(f"\t{old:20}\t->\t{new}")

    if args.quantize_pred:
        df[args.pred] = df[args.pred].mod(args.quantize_pred)

    log(f"metric is {args.metric}...")

    if args.preload and (not args.batchsize or args.batchsize < len(df)):
        log(f"WARNING: received --preload but not --batchsize. Preloaded data will be discarded when another batch starts.")

    ##############
    # MONITORING #
    ##############

    log(f"starting...")
    last_update = time.time()

    if args.parallelize > 1 and args.backend != 'bf':
        assert args.metric == "infinity", f"parallelization only implemented for infinity metric"
        start_time = -1
        result_queue = Queue()
        processes = []
        metrics = {}
        n_processed = 0
        n_flagged = 0
        timings = []
        last_item_time = time.time()
        subresults = defaultdict(lambda: defaultdict(list))  # {iteration: {worker: result}}
        worker_progress = {p : 0 for p in range(args.parallelize)}
        monitor_positives = [] # combined results
        partitions = partition(df, args.parallelize, args.pred)

        for idx, partition in enumerate(partitions):
            p = Process(target=process_partition, args=(args, partition, idx, result_queue))
            processes.append(p)
            p.start()

        while any(p.is_alive() for p in processes) or not result_queue.empty():
            #debug(f"master: loop cond: {any(p.is_alive() for p in processes)} {not result_queue.empty()}")
            if time.time() - last_update > 1:
                log(f"\tprocessed {n_processed}, in queue: ~{result_queue.qsize()}")
                last_update = time.time()

            try:
                msg_type, worker_id, payload = result_queue.get(timeout=1)

                if msg_type == "metrics": # sent upon finishing
                    metrics[f"worker_{worker_id}"] = payload

                elif msg_type == "iter":
                    if start_time == -1: 
                        log(f"\tfirst worker {worker_id} ready")
                        start_time = time.time() # Start time is when first worker is ready
                    worker_completed_iter, cexs = payload
                    worker_progress[worker_id] = worker_completed_iter
                    debug(f"master: recv {(msg_type, worker_id, payload)}, worker_progress: {worker_progress}")
                    for low_id, high_id in cexs:
                        subresults[high_id][worker_id].append(low_id)

                    while n_processed <= min(worker_progress.values()): # Aggregate subresults
                        cexs = set(subresults[n_processed][0]) # compute intersection of all workers' results (Linf only)
                        for worker in range(1, args.parallelize):
                            cexs &= set(subresults[n_processed][worker])
                        for cex in cexs: # collect
                            monitor_positives.append([cex, n_processed])

                        timings.append(time.time() - last_item_time)
                        debug(f"master: {n_processed} complete with cexs {cexs}")
                        if cexs: n_flagged += 1
                        n_processed += 1
                        last_item_time = time.time()
            except Empty:
                continue

        # done, collect overall stats
        metrics['n_processed'] = n_processed
        metrics['n_flagged'] = n_flagged
        metrics['total_time'] = time.time() - start_time
        metrics['args'] = vars(args),
        metrics['perc_flagged'] = n_flagged / n_processed
        metrics['avg_time'] = (time.time() - start_time) / n_processed
        metrics['peak_mem'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        if args.full_output:
            metrics['timings'] = [round(t, 6) for t in timings]

    else:
        runner = setup_backend(args, df)
        start_time = time.time()

        monitor_positives = []
        for i, cexs in enumerate(runner.run(df, args.n_examples, max_time=args.max_time)):
            monitor_positives.extend(cexs)
            if time.time() - last_update > 1:
                log(f"\tprocessed {i}")
                last_update = time.time()

        monitor_positives.sort()
        metrics = collect_metrics(runner, args)

    log(f"done. found {len(monitor_positives)} pairs in {time.time() - start_time:.2f}s")

    ##########
    # OUTPUT #
    ##########

    if args.verbose:
        for pair in monitor_positives:
            if blind_df is not None:
                pretty_print(blind_df, pair[0], pair[1], args.eps, marker="*")
                pretty_print(df, pair[0], pair[1], args.eps, header=False)
            else:
                pretty_print(df, pair[0], pair[1], args.eps)


    if args.out_path:
        if not args.concise_output:
            metrics['positives'] = [(int(x), int(y)) for x, y in monitor_positives]
        with open(args.out_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    ##########################
    # DIFF/CONSISTENCY CHECK #
    ##########################

    if args.diff:
        with open(args.diff) as f:
            this_run_counts = defaultdict(lambda: 0)
            other_run_counts = defaultdict(lambda: 0)
            all_labels = set()
            data = json.load(f)
            other_run = data['positives']
            for x,y in monitor_positives:
                this_run_counts[x] += 1
                this_run_counts[y] += 1
                all_labels.add(x)
                all_labels.add(y)
            for x,y in other_run:
                other_run_counts[x] += 1
                other_run_counts[y] += 1
                all_labels.add(x)
                all_labels.add(y)
            if all_labels:
                print("DIFF:", f"      id", "#this", "gt/lt", "#other", sep="\t")
                for label in sorted(all_labels):
                    if label not in this_run_counts:
                        print("", f"{label:8}", "-", "<", other_run_counts[label], sep="\t")
                    elif label not in other_run_counts:
                        print("", f"{label:8}", this_run_counts[label], ">", "-", sep="\t")
                    elif this_run_counts[label] != other_run_counts[label]:
                        chev = "<" if this_run_counts[label] < other_run_counts[label] else ">"
                        print("", f"{label:8}", this_run_counts[label], chev, other_run_counts[label], sep="\t")

    if args.pairwise_diff:
        with open(args.pairwise_diff) as f:
            data = json.load(f)
            this_run = list(map(lambda tup: str(list(tup)), monitor_positives))
            other_run = list(map(lambda tup: str(list(tup)), data['positives']))
            rqks = data.get('radius_query_ks', [])
            seen = defaultdict(lambda: 0)
            for positive in this_run:
                seen[positive] += 1
            for positive in other_run:
                seen[positive] += 1
            this_only = []
            other_only = []
            both = []
            for k,v in seen.items():
                if v != 2: # not seen by both
                    if k in this_run:
                        this_only.append(k)
                    else:
                        other_only.append(k)
                else:
                    both.append(k)
            first_id = lambda x: int(x.split(",")[0][1:])
            for label, data in (("This run", sorted(this_only, key=first_id)), 
                                (args.pairwise_diff, sorted(other_only, key=first_id)),
                                ("Both runs", sorted(both, key=first_id))):
                print(f"#################\n{label}:", data, sep='\t')
                for pair in map(lambda s: json.loads(s), data):
                    if len(rqks) > 1:
                        print(pair, "radius query k =", rqks[pair[1]])
                    else:
                        print(pair)
                    pretty_print(df, pair[0], pair[1], args.eps, header=False, metric=args.metric)
