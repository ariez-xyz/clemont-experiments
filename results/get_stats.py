import json
import sys
import os
from pathlib import Path

results = []
directory = Path(sys.argv[1])

# Find and process all JSON files in the directory
for json_file in directory.glob('**/*.json'):
    with open(json_file) as f:
        data = json.load(f)
        print(json_file, file=sys.stderr)
        if type(data['args']) == list:
            data['args'] = data['args'][0]
        if type(data['peak_mem']) == list:
            data['peak_mem'] = data['peak_mem'][0]
        
        # Extract the basename from the csvpath
        csvpath = data['args']['out_path']
        basename = os.path.basename(csvpath)

        if 'parallelize' in data['args'].keys():
            backend = f"{data['args']['backend']}@{data['args']['parallelize']}t"
        else:
            backend = data['args']['backend']

        i = 0
        worker_mem_deltas = []
        while f'worker_{i}' in data.keys():
            worker_mem_deltas.append(data[f'worker_{i}']['peak_mem'] - data[f'worker_{i}']['mem'][0]/1024)
            i += 1
        #if worker_mem_deltas:
        #    mem_delta = sum(worker_mem_deltas)/len(worker_mem_deltas)
        #else:
        #    mem_delta = data['peak_mem'] - data['mem'][0]/1024

        # Create tuple with required fields
        result_tuple = (
            basename,
            data['args']['eps'],
            data['avg_time'],
            backend,
            #round(mem_delta),
            data['peak_mem'],
            len(data['positives']),
        )
        results.append(result_tuple)

# Sort and print results
#print("filename, eps, avg time, backend, memory delta between first and last iter (averaged per-worker if parallelized), n_positives")
#print("filename, eps, avg time, backend, peak memory recorded by main process, n_positives")
for result in sorted(results):
    print(result)

print("WARNING: Current script is not counting memory usage from worker processes if there are any", file=sys.stderr)
