import os
import json
import glob

def count_unique_second_elements(tuples_list):
    return len(set(t[1] for t in tuples_list))

def get_dataset_size(dataset_name, source):
    if source == 'certifair':
        sizes = {
            'adult': 46034,
            'compas': 5279,
            'german': 1001
        }
    else:  # lcifr
        sizes = {
            'adult': "TODO",
            'compas': 4223,
            'german': 801
        }
    return sizes[dataset_name]

def process_files():
    results = []
    
    # Process Certifair files (only P2)
    certifair_files = glob.glob('./results/certifair/*-P2-*.json')
    for filepath in certifair_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        filename = os.path.basename(filepath)
        dataset = filename.split('-')[0]
        mode = filename.split('-')[1]
        eps = filename.split('eps')[1].replace('.json', '')
        
        unique_count = count_unique_second_elements(data['positives'])
        dataset_size = get_dataset_size(dataset, 'certifair')
        percentage = (unique_count / dataset_size) * 100
        
        results.append((f"Certifair {dataset.title()} {mode} P2 eps={eps}", percentage))
    
    # Process LCIFR files
    lcifr_files = glob.glob('./results/lcifr/*.json')
    for filepath in lcifr_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        filename = os.path.basename(filepath)
        parts = filename.split('_')[1:]  # Skip 'predictions'
        dataset = parts[0]
        dl2 = parts[1].split('-')[1]  # Extract dl2 value
        eps = filename.split('eps')[1].replace('.json', '')
        
        unique_count = count_unique_second_elements(data['positives'])
        dataset_size = get_dataset_size(dataset, 'lcifr')
        percentage = (unique_count / dataset_size) * 100
        
        results.append((f"Lcifr {dataset.title()} Dl2-{dl2} eps={eps}", percentage))
    
    # Sort by percentage and print
    results.sort()
    for description, percentage in results:
        print(f"{description} {percentage:.2f}%")

if __name__ == "__main__":
    process_files()
