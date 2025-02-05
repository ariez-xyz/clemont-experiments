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
        
        # Extract the basename from the csvpath
        csvpath = data['args'][0]['csvpath'][0]
        basename = os.path.basename(csvpath)
        
        # Create tuple with required fields
        result_tuple = (
            basename,
            data['args'][0]['eps'],
            data['avg_time'],
            data['peak_mem'][0]
        )
        results.append(result_tuple)

# Sort and print results
for result in sorted(results):
    print(result)
