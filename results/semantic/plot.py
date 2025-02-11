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

    # Drop bit_error if present
    if 'bit_error' in df.columns:
        df = df.drop('bit_error', axis=1)

    # Define semantic groupings and sort order
    imagenet_c_order = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',  # noise group
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # blur group
        'snow', 'frost', 'fog',  # weather group
        'brightness', 'contrast',  # intensity group
        'elastic_transform', 'pixelate', 'jpeg_compression'  # digital group
    ]

    camera_order = [
        'iso_noise',  # noise
        'near_focus', 'far_focus',  # focus
        'xy_motion_blur', 'z_motion_blur',  # motion
        'flash', 'low_light',  # lighting
        'color_quant'  # digital
    ]

    # Apply the appropriate sort order
    if all(col in df.columns for col in imagenet_c_order):
        df = df[imagenet_c_order]
    elif all(col in df.columns for col in camera_order):
        df = df[camera_order]
    
    return df, epsilon, metric

def process_all_directories(base_dir='results', use_rate=False, outfile="out.png"):
    # Find all model directories
    subdirs = sorted([d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)])
    is_imagenet = any('imagenet' in d for d in subdirs)
    if is_imagenet: subdirs = reversed(subdirs)
    
    # Collect data from all directories
    model_data = {}
    global_vmin = 0#float('inf')
    global_vmax = 1.52#float('-inf')
    
    for subdir in subdirs:
        if is_imagenet and '7.5' in subdir: continue
        df, eps, metric = process_directory(subdir, use_rate)
        if not df.empty:
            model_name = os.path.basename(subdir)
            # Transpose the dataframe here
            df = df.T
            if use_rate:
                values = np.log10(df * 100 + 1)
            else:
                values = np.log10(df.clip(lower=1))
                
#            global_vmin = min(global_vmin, values.values.min())
#            global_vmax = max(global_vmax, values.values.max())
            model_data[model_name] = (df, eps, metric)
    
    if not model_data:
        return
        
    n_models = len(model_data)
    # Adjust figure size - make it a bit taller to accommodate colorbar
    if is_imagenet:
        fig = plt.figure(figsize=(0.2+n_models, 5))
    else:
        fig = plt.figure(figsize=(1+n_models, 5))
    
    # Create grid with more space for colorbar
    gs = plt.GridSpec(2, n_models, height_ratios=[20, 1])
    
    axes = []
    for idx, (model_name, (df, eps, metric)) in enumerate(model_data.items()):
        ax = fig.add_subplot(gs[0, idx])
        axes.append(ax)
        
        show_yticks = (idx == 0)
        # Further simplify model names
        short_name = 'Tian' if 'Tian' in model_name else model_name.split('-')[-1].split('2')[0]
        
        plot_heatmap(df, short_name, ax=ax, 
                    vmin=global_vmin, vmax=global_vmax,
                    show_yticks=show_yticks,
                    use_rate=use_rate)
    
    # Add shared colorbar with logarithmic scale ticks
    norm = plt.Normalize(vmin=10**global_vmin, vmax=10**global_vmax)
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    
    # Values we want to show
    tick_values = np.array([0, 1.42, 4.83, 13.09, 33.03])
    
    # Calculate positions that are equally spaced
    tick_positions = np.linspace(10**global_vmin, 10**global_vmax, len(tick_values))
    
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([f'{x:.2f}' for x in tick_values])
    
    cbar.set_label('% violations', labelpad=10)

    
    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.05)
    
    plt.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close()

def plot_heatmap(df, title, ax=None, vmin=None, vmax=None, show_yticks=True, use_rate=False):
    if use_rate:
        plot_data = np.log10(df * 100 + 1)
    else:
        plot_data = np.log10(df.clip(lower=1))
    
    sns.heatmap(plot_data, 
                ax=ax,
                cmap='YlOrRd',
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                )#square=True)
    
    ax.set_title(title, pad=5, size=10)
    ax.set_xlabel('Severity', labelpad=5)
    if not show_yticks:
        ax.set_yticks([])
    
    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_xticklabels(range(1, 6))

def main():
    parser = argparse.ArgumentParser(description='Process and visualize corruption test results')
    parser.add_argument('directory', type=str, default='results', 
                        help='Directory containing JSON result files (optional)')
    parser.add_argument('--rate', action='store_true', 
                        help='Use true positive rate instead of absolute numbers')
    parser.add_argument('--out', type=str, default='comparison_heatmap.png',
                        help='outfile')
    args = parser.parse_args()
    
    process_all_directories(base_dir=args.directory, use_rate=args.rate, outfile=args.out)

if __name__ == "__main__":
    main()


