import json
import glob
import os
import sys
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
    parser.add_argument('--sample', type=str, default="11:",
                       help='dimensionality to plot')
    parser.add_argument('--eps', type=str, default="0.025",
                       help='epsilon to plot')
    parser.add_argument('--outfile', type=str, default="fig.png",
                       help='name of file to save to')
    parser.add_argument('--omit_beginning', type=int, default=100,
                       help='omit this many samples at the beginning (avoid startup cost going into rolling average)')
    parser.add_argument('--run', type=int, default=1,
                       help='-run$i postfix to use')
    return parser.parse_args()

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    method = os.path.basename(os.path.dirname(filepath))
    if args.run == 1:
        norm, eps, parallelization, sample = filename.replace('.json', '').split('-')
    else:
        norm, eps, parallelization, sample, run = filename.replace('.json', '').split('-')
    
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

    name = f"{method_displaynames[method]} ({norm_displaynames[norm]})"
    if parallelization != '1':
        name += f'{parallelization} threads'
    
    return {
        'method': method,
        'batchsize': 10000,#int(batchsize),
        'norm': norm,
        'eps': float(eps),
        'name': name,
        'parallelization': parallelization,
    }

def rolling_average(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

args = parse_args()

# Read all JSON files
data = []
for filepath in glob.glob(os.path.join(args.results_dir, 'results/*/*.json')):
    if (args.run == 1 and 'run' in filepath) or (args.run > 1 and f'run{args.run}' not in filepath):
        continue
    if args.eps not in filepath or args.sample not in filepath:
        continue
    with open(filepath, 'r') as f:
        print("adding", filepath, file=sys.stderr)
        result = json.load(f)
        file_info = parse_filename(filepath)
        #if args.sample == "11:" and file_info['parallelization'] != '1':
        #    continue
        data.append({**file_info, 'timings': result['timings'][:args.truncate]})

# Sort data by norm, batchsize, method
data.sort(key=lambda x: (x['parallelization'], x['norm'], x['method']))

# Create plot
plt.figure(figsize=(5, 3))

for item in data:
    x = np.arange(len(item['timings'][args.omit_beginning:]))[::1000]
    y = rolling_average(item['timings'][args.omit_beginning:], args.windowsize)[::1000]
    plt.plot(x, y, '-' if item['paralleliation'] == '1' else '--', label=item["name"])

plt.xlabel('Sample')
plt.ylabel('Time (seconds)')
plt.ylim(top=0.07)
plt.grid(True, alpha=0.3)
#plt.title(f"dropcols={args.sample}, eps={args.eps}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(args.results_dir, args.outfile), dpi=300)
