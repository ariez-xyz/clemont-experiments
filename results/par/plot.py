import os
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_dim_from_filename(filename):
    return int(filename.split('-')[0])

def get_average_time(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data['avg_time']
        #timings = data['timings']
        #if type(timings[0]) == str:
        #    timings = list(map(float,timings))
        #return np.mean(timings)

results_dir = 'results'
methods = os.listdir(results_dir)

plt.figure(figsize=(5, 4))

y_min = 0.001
y_max = 1.0

# Use matplotlib's default colors
colors = {
    'kdtree': '#1f77b4',  # matplotlib default blue
    'bdd': '#ff7f0e'      # matplotlib default orange
}

line_styles = {
    '1t': '-',
    '16t': '--',
    '96t': ':'
}

for method in methods:
    method_dir = os.path.join(results_dir, method)
    json_files = os.listdir(method_dir)
    
    dimensions = []
    avg_times = []
    
    for json_file in json_files:
        dim = extract_dim_from_filename(json_file)
        avg_time = get_average_time(os.path.join(method_dir, json_file))
        
        dimensions.append(dim)
        avg_times.append(avg_time)
    
    sorted_indices = np.argsort(dimensions)
    dimensions = np.array(dimensions)[sorted_indices]
    avg_times = np.array(avg_times)[sorted_indices]
    
    # Mask out points outside the y-axis limits
    mask = (avg_times >= y_min) & (avg_times <= y_max)
    
    # Determine color and line style
    base_method = 'kdtree' if 'kdtree' in method else 'bdd'
    thread_style = method.split('-')[1]  # Gets '1t', '16t', or '96t'
    
    method_displaynames = {
        "kdtree-1t": "Kd-tree (1 thread)",
        "kdtree-16t": "Kd-tree (16 threads)",
        "kdtree-96t": "Kd-tree (96 threads)",
        "bdd-1t": "BDD (1 thread)",
        "bdd-16t": "BDD (16 threads)",
        "bdd-96t": "BDD (96 threads)",
    }
    
    # Plot line for all points
    plt.plot(dimensions, avg_times, 
             line_styles[thread_style], 
             color=colors[base_method],
             label=method_displaynames[method])
    # Plot markers only for points within limits
    plt.plot(dimensions[mask], avg_times[mask], 'o', 
             color=colors[base_method])

plt.xscale('log', base=2)
plt.yscale('log')
plt.ylim(y_min, y_max)
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Processing Time (seconds)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('dimensions.png', dpi=300)
plt.close()

