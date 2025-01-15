import argparse
import torch
import pandas as pd
import time
from datetime import datetime
from robustbench.data import load_cifar10
from robustbench.utils import load_model

def make_argparser():
    parser = argparse.ArgumentParser(description='Run inference with RobustBench models')
    parser.add_argument('--model', type=str, default='Standard',
                        help='Name of the model to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar10c', 'cifar100', 'cifar100c', 'imagenet', 'imagenet3dcc'],
                        help='Dataset to use')
    parser.add_argument('--threat-model', type=str, default='Linf',
                        choices=['Linf', 'L2', 'corruptions'],
                        help='Threat model for robust training')
    parser.add_argument('--n-examples', '--n_examples', type=int, default=-1,
                        help='Number of examples to process')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    return parser

def load_dataset(name, n_examples):
    """Notes on datasets: ImageNet must be downloaded manually. 
    See https://github.com/RobustBench/robustbench?tab=readme-ov-file#model-zoo
    """

    if n_examples < 0:
        n_examples = None # RobustBench will load the full dataset for n_examples=None
    if name == 'cifar10':
        x_test, y_test = load_cifar10(n_examples=n_examples)
    elif name == 'cifar100':
        x_test, y_test = load_cifar100(n_examples=n_examples)
    else:
        raise ValueError(f"unsupported dataset {dataset}")
    return x_test, y_test

def log(s):
    timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {s}", flush=True)

def classify(model, data):
    predictions = []
    last_update = time.time()

    with torch.no_grad():
        for i, x in enumerate(data):
            x = x.unsqueeze(0)  # Add batch dimension (x.shape 3,32,32 -> 1,3,32,32)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred)

            if time.time() - last_update > 1:
                log(i)
                last_update = time.time()

    return predictions

if __name__ == '__main__':
    args = make_argparser().parse_args()
    log(f"Model: {args.model}, Dataset: {args.dataset}, Threat model: {args.threat_model}, Number of examples: {args.n_examples}, Output path: {args.output}")

    model = load_model(model_name=args.model, dataset=args.dataset, threat_model=args.threat_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # no grad descent

    x_test, _ = load_dataset(args.dataset, args.n_examples)
    x_test = x_test.to(device)
    log(f"{args.model} model and {len(x_test)} samples from {args.dataset} loaded on device {device}")

    predictions = [i.item() for i in classify(model, x_test)]

    # Flatten, e.g. x.shape becomes n,3,32,32 -> n,3072
    x_test = x_test.reshape(len(x_test), -1)

    # save csv
    df = pd.DataFrame({
        'pred': predictions,
        **{f'f{i}': x_test[:, i].cpu().numpy() for i in range(x_test.shape[1])}
    })
    df.to_csv(args.output, index=False)
    log("completed")

