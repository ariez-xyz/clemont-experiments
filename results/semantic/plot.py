import json
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import re

warned_missing_datafiles = False
def warn(s):
    global warned_missing_datafiles
    if not warned_missing_datafiles:
        warned_missing_datafiles = True
        print(s)

def parse_filename(filename):
    base = os.path.basename(filename)
    if "vanilla" in base:
        return None, None
    
    pattern = r'.*-([a-zA-Z_]+)-(\d+)\.json$'
    match = re.match(pattern, base)
    if match:
        corruption, severity = match.groups()
        return corruption, int(severity)
    return None, None

def get_total_size(csvpaths):
    try:
        return sum(sum(1 for _ in open(csv_path)) - 1 for csv_path in csvpaths)
    except FileNotFoundError:
        warn(f"WARNING: could not find original data csv's - guessing length.")
        if "imagenet" in csvpaths[0].lower():
            baselen = 5000
        baselen = 10000
        return len(csvpaths) * baselen

def get_values_from_json(json_file, use_rate=False):
    with open(json_file, 'r') as f:
        data = json.load(f)
        if not use_rate:
            val = data['n_true_positives']
        else:
            # Get unique second elements from positives list
            unique_second_ids = len(set(p[1] for p in data['positives']))
            total_size = get_total_size(data['args']['csvpath'])
            val = unique_second_ids / total_size
        eps = data['eps']
        metric = data['args']['metric']
        return val, eps, metric

def process_directory(directory_path, use_rate=False):
    results = {}
    
    for json_file in glob.glob(os.path.join(directory_path, '*.json')):
        corruption, severity = parse_filename(json_file)
        if corruption is None:
            continue
            
        value, epsilon, metric = get_values_from_json(json_file, use_rate)
        
        if corruption not in results:
            results[corruption] = {}
        results[corruption][severity] = value
    
    df = pd.DataFrame.from_dict({k: pd.Series(v) for k, v in results.items()})
    df.index.name = 'Severity'
    df = df.reindex(sorted(df.columns), axis=1)
    
    return df, epsilon, metric

def plot_heatmap(df, title, save_path=None, vmin=None, vmax=None):
    plt.figure(figsize=(5, 2.6), dpi=300)

    df = df.iloc[::-1]  # Reverse the order of rows
    
    is_rate = df.values.max() <= 1
    if is_rate:
        plot_data = np.log10(df * 100 + 1)
        raw_data = df * 100
    else:
        plot_data = np.log10(df.clip(lower=1))
        raw_data = df
        
    if vmin is None:
        vmin = plot_data.values.min()
    if vmax is None:
        vmax = plot_data.values.max()
    
    # Create the heatmap
    hm = sns.heatmap(plot_data, 
                     # Uncomment to add labels to each cell.
                     #annot=raw_data, 
                     cmap='YlOrRd',
                     fmt='.2f' if is_rate else 'g',
                     vmin=vmin,
                     vmax=vmax,
                     cbar_kws={'format': '%.2f'})
    
     # Uncomment to add a title.
    #plt.title(title)
    plt.ylabel('Severity')
    plt.xlabel('Corruption Type')
    plt.xticks(rotation=45, ha='right')
    
    # Modify colorbar to show actual values instead of log values
    cbar = hm.collections[0].colorbar
    
    # Calculate tick positions in log space
    if is_rate:
        tick_locations = np.linspace(vmin, vmax, 5)
        tick_labels = np.round(100 * (10 ** tick_locations - 1) / 100, 2)
    else:
        tick_locations = np.linspace(vmin, vmax, 5)
        tick_labels = np.round(10 ** tick_locations, 2)
    
    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Percentage' if is_rate else 'Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def process_all_directories(base_dir='results', use_rate=False):
    subdirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    
    all_dfs = []
    all_eps = []
    all_metrics = []
    for subdir in subdirs:
        df, eps, metric = process_directory(subdir, use_rate)
        if not df.empty:
            all_dfs.append(df)
            all_eps.append(eps)
            all_metrics.append(metric)
    
    if use_rate:
        global_vmin = min(np.log10(df * 100 + 1).values.min() for df in all_dfs)
        global_vmax = max(np.log10(df * 100 + 1).values.max() for df in all_dfs)
    else:
        global_vmin = min(np.log10(df.clip(lower=1)).values.min() for df in all_dfs)
        global_vmax = max(np.log10(df.clip(lower=1)).values.max() for df in all_dfs)
    
    for subdir, df, eps, metric in zip(subdirs, all_dfs, all_eps, all_metrics):
        dirname = os.path.basename(subdir)
        title = f"{'%' if use_rate else '#'} flagged inputs by Corruption Type and Severity\n{dirname}, eps={eps}, metric={metric}"
        save_path = f"{dirname}_heatmap.png"
        plot_heatmap(df, title, save_path, vmin=global_vmin, vmax=global_vmax)

def main():
    parser = argparse.ArgumentParser(description='Process and visualize corruption test results')
    parser.add_argument('directory', type=str, nargs='?', 
                        help='Directory containing JSON result files (optional)')
    parser.add_argument('--rate', action='store_true', 
                        help='Use true positive rate instead of absolute numbers')
    args = parser.parse_args()
    
    if args.directory:
        if not os.path.isdir(args.directory):
            print(f"Error: {args.directory} is not a valid directory")
            return
        
        df = process_directory(args.directory, args.rate)
        if not df.empty:
            dirname = os.path.basename(args.directory)
            title = f"{'True Positive Rate' if args.rate else 'True Positives'} by Corruption Type and Severity\n{dirname}"
            plot_heatmap(df, title)
    else:
        process_all_directories(use_rate=args.rate)

if __name__ == "__main__":
    main()


