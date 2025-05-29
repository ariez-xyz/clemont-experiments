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
            'adult': 45222,
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
        
        results.append(("Certifair", dataset.title(), f"P2 {mode}", eps, percentage))
    
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
        
        results.append(("Lcifr", dataset.title(), f"Dl2 {dl2}", eps, percentage))
    
    # Sort by percentage and print
    results.sort()
    for p in "Lcifr Certifair".split():
        print(p)
        for d in "Adult Compas German".split():
            print("\t", d)
            for e in "0.0025 0.005 0.01 0.02 0.04 0.08 0.12 0.16 0.2 0.24 0.28 0.32".split():
                print("\t","\t",  e)
                for paper, dataset, fairness, eps, percentage in results:
                    #if paper == p and dataset == d and eps == e:
                    if dataset == d and eps == e:
                        print("\t", "\t", "\t", f"{paper:10}\t{fairness:10}\t{percentage:.2f}%")

if __name__ == "__main__":
    process_files()
