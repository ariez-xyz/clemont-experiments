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
methods = sorted(os.listdir(results_dir))

plt.figure(figsize=(5, 3))

y_min = 0.0001
y_max = 0.8


colors = {
    'bdd-1t': '#4c9edb',     # light blue
    'bdd-16t': '#1f77b4',    # default blue
    'bdd-96t': '#164b73',    # dark blue
    'kdtree-1t': '#ffb74d',  # light orange
    'kdtree-16t': '#f57c00', # default orange
    'kdtree-96t': '#b35a00', # dark orange
}

line_styles = {
    '1t': '-',
    '16t': '-',
    '96t': '-'
}

for method in methods:
    method_dir = os.path.join(results_dir, method)
    json_files = os.listdir(method_dir)
    
    dimensions = []
    avg_times = []
    
    for json_file in json_files:
        dim = extract_dim_from_filename(json_file)
        avg_time = get_average_time(os.path.join(method_dir, json_file))
        
        if dim > 2:
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
             color=colors[f'{base_method}-{thread_style}'],
             label=method_displaynames[method])
    # Plot markers only for points within limits
    plt.plot(dimensions[mask], avg_times[mask], {'1t': 'o', '16t': 'x', '96t': 'v'}[thread_style], 
             color=colors[f'{base_method}-{thread_style}'])

plt.xscale('log', base=2)
plt.yscale('log')
plt.ylim(top=y_max)
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Processing Time (seconds)')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
even_powers = np.arange(2, 17, 2)
plt.xticks(2**even_powers, [f'$2^{{{p}}}$' for p in even_powers])
plt.tight_layout()
plt.savefig('parallelization.png', dpi=300)
plt.close()

