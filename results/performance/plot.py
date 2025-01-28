import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Plot timing results from JSON files')
    parser.add_argument('--results-dir', type=str, default=os.getcwd(),
                       help='Directory containing results (default: current directory)')
    parser.add_argument('--windowsize', type=int, default=100000,
                       help='rolling average windowsize')
    parser.add_argument('--fix-batchsize', type=int, default=10000,
                       help='Fix batch size to a specific value')
    parser.add_argument('--truncate', type=int, default=5_000_000,
                       help='disregard samples past a certain point')
    parser.add_argument('--omit_beginning', type=int, default=100,
                       help='omit this many samples at the beginning (avoid startup cost going into rolling average)')
    return parser.parse_args()

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    method = os.path.basename(os.path.dirname(filepath))
    batchsize, norm, eps = filename.replace('.json', '').split('-')
    
    norm_displaynames = {
        "infinity": "Linf",
        "l2": "L2",
    }
    method_displaynames = {
        "kdtree": "Kd-tree",
        "bdd": "BDD",
        "snn": "SNN",
        "bf": "Brute force",
    }

    name = f"{method_displaynames[method]}, {norm_displaynames[norm]} ε={eps}"
    if batchsize != '0':
        name += f" (batchsize={int(batchsize)//1000}k)"
    
    return {
        'method': method,
        'batchsize': int(batchsize),
        'norm': norm,
        'eps': float(eps),
        'name': name
    }

def rolling_average(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

args = parse_args()

# Read all JSON files
data = []
for filepath in glob.glob(os.path.join(args.results_dir, 'results/*/*.json')):
    with open(filepath, 'r') as f:
        result = json.load(f)
        file_info = parse_filename(filepath)
        data.append({**file_info, 'timings': result['timings'][:args.truncate]})

# Sort data by norm, batchsize, method
data.sort(key=lambda x: (x['norm'], x['method'], x['eps'], x['batchsize']))
if args.fix_batchsize != None:
    data = list(filter(lambda x: int(x['batchsize']) == args.fix_batchsize or str(x['batchsize']) == '0', data))

# Create plot
plt.figure(figsize=(7.1, 5))

for item in data:
    x = np.arange(len(item['timings']))
    y = rolling_average(item['timings'], args.windowsize)
    plt.plot(x[args.omit_beginning:], y[args.omit_beginning:], label=item["name"])

plt.xlabel('Sample')
plt.ylabel('Time (seconds)')
if '12d' in args.results_dir:
    plt.title(f'Processing Time per Sample, 5M rows, 12 cols, {args.windowsize // 1000}k rolling average')
else:
    plt.title(f'Processing Time per Sample, 5M rows, 22 cols, {args.windowsize // 1000}k rolling average')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.results_dir, 'fig.png'), dpi=300)
