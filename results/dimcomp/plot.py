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
    
    method_displaynames = {
        "kdtree": "Kd-tree (Linf)",
        "l2-kdtree": "Kd-tree (L2)",
        "bdd": "BDD (Linf)",
        "l2-snn": "SNN (L2)",
        "bf": "Brute force (Linf)",
    }
    plt.plot(dimensions, avg_times, marker='o', label=method_displaynames[method])
    if method == 'bdd':
        plt.plot([2**6, 2**7], [avg_times[-1], 100], color='#1f77b4', linestyle='--', alpha=0.5)
    if method == 'l2-snn':
        plt.plot([2**15, 2**16], [avg_times[-1], 25], color='#ff7f0e', linestyle='--', alpha=0.5)

plt.xscale('log', base=2)
plt.yscale('log')
plt.ylim(top=4)
plt.xlabel('Number of Dimensions')
plt.ylabel('Average Processing Time (seconds)')
#plt.title('Processing Time vs Dimensions by Method')
plt.grid(True, which="both", ls="-", alpha=0.5)



plt.legend()
plt.tight_layout()
plt.savefig('dimensions.png', dpi=300)
plt.close()
