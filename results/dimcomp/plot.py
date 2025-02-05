import os
import json
import matplotlib.pyplot as plt
import numpy as np

def extract_dim_from_filename(filename):
    return int(filename.split('-')[0])

def get_average_time(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        timings = data['timings']
        if type(timings[0]) == str:
            timings = list(map(float,timings))
        return np.mean(timings)

results_dir = 'results'
methods = os.listdir(results_dir)

plt.figure(figsize=(5, 4))

for method in methods:
    method_dir = os.path.join(results_dir, method)
    json_files = os.listdir(method_dir)
    
    dimensions = []
    avg_times = []
    
    for json_file in json_files:
        print("PLOTTING ONLY LOWDEC FILES")
        if 'lowdec' in json_file and not "bf-1t" in json_file:
            dim = extract_dim_from_filename(json_file)
            avg_time = get_average_time(os.path.join(method_dir, json_file))
            
            dimensions.append(dim)
            avg_times.append(avg_time)
    
    sorted_indices = np.argsort(dimensions)
    dimensions = np.array(dimensions)[sorted_indices]
    avg_times = np.array(avg_times)[sorted_indices]
    
    method_displaynames = {
        "kdtree-1t": "Kd-tree (Linf)",
        "kdtree-16t": "Kd-tree (Linf, 16 threads)",
        "kdtree-96t": "Kd-tree (Linf, 96 threads)",
        "kdtree-l2": "Kd-tree (L2)",
        "bdd-1t": "BDD",
        "bdd-16t": "BDD (16 threads)",
        "bdd-96t": "BDD (96 threads)",
        "snn": "SNN",
        "bf-1t": "Brute force",
        "bf-16t": "Brute force (16 threads)",
        "bf-96t": "Brute force (96 threads)",
    }
    plt.plot(dimensions, avg_times, marker='o', label=method_displaynames[method])

plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Processing Time (seconds)')
#plt.title('Processing Time vs Dimensions by Method')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('dimensions.png', dpi=300)
plt.close()
