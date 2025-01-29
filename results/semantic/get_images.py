import json
import random
import os
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd
from typing import Optional
from robustbench.data import load_cifar10, load_cifar100, load_cifar10c, load_cifar100c, load_imagenet, load_imagenet3dcc
from torchvision import datasets


def load_dataset(name, n_examples, corruption=None, severity=5):

    if name == 'cifar10':
        x_test, y_test = load_cifar10(n_examples=n_examples, data_dir="../../data/RobustBench/data/")
    elif name == 'cifar100':
        x_test, y_test = load_cifar100(n_examples=n_examples, data_dir="../../data/RobustBench/data/")
    elif name == 'imagenet':
        x_test, y_test = load_imagenet(n_examples=n_examples, data_dir="../../data/RobustBench/data/")
    elif name == 'cifar10c':
        if not corruption: raise ValueError("cifar10c requires specifying a corruption")
        x_test, y_test = load_cifar10c(n_examples=n_examples, corruptions=[corruption], severity=severity, data_dir="../../data/RobustBench/data/")
    elif name == 'cifar100c':
        if not corruption: raise ValueError("cifar100c requires specifying a corruption")
        x_test, y_test = load_cifar100c(n_examples=n_examples, corruptions=[corruption], severity=severity, data_dir="../../data/RobustBench/data/")
    elif name == 'imagenet3dcc':
        if not corruption: raise ValueError("imagenet3dcc requires specifying a corruption")
        x_test, y_test = load_imagenet3dcc(n_examples=n_examples, corruptions=[corruption], severity=severity, data_dir="../../data/RobustBench/data/")
    else:
        raise ValueError(f"unsupported dataset {name}")
    return x_test, y_test

def visualize_positives_from_json(json_path, max_samples):
    # Parse the filename to determine dataset and corruption details
    filename = os.path.basename(json_path)
    corruption = None
    severity = None
    
    model = filename.split('-')[-5]
    if filename.startswith('cifar10c-'):
        dataset_clean = 'cifar10'
        dataset_corrupt = 'cifar10c'
        threshold = 10000
        corruption = filename.split('-')[-2]
        severity = int(filename.split('-')[-1].split('.')[0])
    elif filename.startswith('cifar100c-'):
        dataset_clean = 'cifar100'
        dataset_corrupt = 'cifar100c'
        threshold = 10000
        corruption = filename.split('-')[-2]
        severity = int(filename.split('-')[-1].split('.')[0])
    elif filename.startswith('imagenet3dcc-'):
        dataset_clean = 'imagenet'
        dataset_corrupt = 'imagenet3dcc'
        threshold = 5000
        corruption = filename.split('-')[-2]
        severity = int(filename.split('-')[-1].split('.')[0])
    else:
        raise ValueError(f"Unsupported dataset in filename: {filename}")

    # Load positives from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    positives = data['positives']
    np = len(positives)
    perc = len(set([t[1] for t in positives])) / (2*threshold)
    
    # Sample if too many positives
    if len(positives) > max_samples:
        positives = random.sample(positives, max_samples)
    
    # Load datasets
    x_clean, _ = load_dataset(dataset_clean, n_examples=threshold)
    x_corrupt, _ = load_dataset(dataset_corrupt, n_examples=threshold, corruption=corruption, severity=severity)

    # Load model predictions
    pred_clean = pd.read_csv(f"../../data/RobustBench/predictions/{dataset_corrupt}-{model}/vanilla.csv")["pred"]
    pred_corrupt = pd.read_csv(f"../../data/RobustBench/predictions/{dataset_corrupt}-{model}/{corruption}-{severity}.csv")["pred"]

    # Map numerical predictions to class names
    if dataset_clean == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # cifar100
        class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    pred_clean = pred_clean.map(lambda x: class_names[int(x)])
    pred_corrupt = pred_corrupt.map(lambda x: class_names[int(x)])
    
    def get_data(idx):
        if idx < threshold:
            return x_clean[idx]
        return x_corrupt[idx - threshold]
    
    pairs_per_row=4
    n_rows = len(positives) // pairs_per_row + (1 if len(positives) % pairs_per_row else 0)
    fig, axes = plt.subplots(n_rows, pairs_per_row * 2 + (pairs_per_row - 1), figsize=(3*pairs_per_row + 1.5*(pairs_per_row-1), 3*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (clean_idx, corrupt_idx) in enumerate(positives):
        # Convert tensors to numpy and transpose to (H,W,C)
        clean_img = get_data(clean_idx).permute(1, 2, 0).numpy()
        corrupt_img = get_data(corrupt_idx).permute(1, 2, 0).numpy()
        
        row = i // pairs_per_row
        col = (i % pairs_per_row) * 3
        
        axes[row, col].imshow(clean_img)
        axes[row, col].set_title(f'{pred_clean[clean_idx]}')
        axes[row, col].axis('off')
        
        axes[row, col + 1].imshow(corrupt_img)
        axes[row, col + 1].set_title(f'{pred_corrupt[corrupt_idx-threshold]}')
        axes[row, col + 1].axis('off')
        
        # Add separator
        if col + 2 < axes.shape[1]:
            axes[row, col + 2].axis('off')

    # Hide empty subplots if any
    for i in range(len(positives), pairs_per_row * n_rows):
        row = i // pairs_per_row
        col = (i % pairs_per_row) * 3
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
        if col + 2 < axes.shape[1]:
            axes[row, col + 2].axis('off')

    fig.suptitle(f"Flagged Inputs for semantic robustness on {dataset_corrupt}\nCorruption: {corruption} severity {severity}\nModel: {model}. {np} TP ({perc*100:.2f}%)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'plots/{model}-{corruption}-{severity}.png', dpi=300, bbox_inches='tight')
    plt.show()



def main():
    parser = argparse.ArgumentParser(description='Visualize positive examples from a JSON file.')
    parser.add_argument('json_path', type=str, help='Path to the JSON file containing positive examples')
    parser.add_argument('--max_samples', type=int, default=12, help='Maximum number of samples to visualize')
    
    args = parser.parse_args()
    
    visualize_positives_from_json(args.json_path, max_samples=args.max_samples)


if __name__ == '__main__':
    main()
