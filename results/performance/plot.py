import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_filename(filepath):
    filename = os.path.basename(filepath)
    method = os.path.basename(os.path.dirname(filepath))
    batchsize, norm, eps = filename.replace('.json', '').split('-')
    
    name = f"{method.upper()}, {norm}, eps={eps}"
    if batchsize != '0':
        name += f" (batchsize={batchsize})"
    
    return {
        'method': method,
        'batchsize': int(batchsize),
        'norm': norm,
        'eps': float(eps),
        'name': name
    }

def rolling_average(data, window):
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

# Read all JSON files
data = []
for filepath in glob.glob('results/snn/*.json'):
    with open(filepath, 'r') as f:
        result = json.load(f)
        file_info = parse_filename(filepath)
        data.append({**file_info, 'timings': result['timings']})

# Sort data by norm, batchsize, method
data.sort(key=lambda x: (x['norm'], x['method'], x['eps'], x['batchsize']))
data = list(filter(lambda x: str(x['batchsize']) == '10000', data))

# Create plot
plt.figure(figsize=(12, 8))

for item in data:
    x = np.arange(len(item['timings']))
    y = rolling_average(item['timings'], 150000)
    plt.plot(x, y, label=item['name'])

plt.xlabel('Sample')
plt.ylabel('Time (seconds)')
plt.title('Processing Time per Sample')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('fig.png', bbox_inches='tight', dpi=300)
