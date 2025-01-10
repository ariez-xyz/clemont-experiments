import argparse
import torch
import pandas as pd
from robustbench.data import load_cifar10
from robustbench.utils import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with RobustBench models')
    parser.add_argument('--model', type=str, default='Carmon2019Unlabeled',
                        help='Name of the model to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar10c', 'cifar100', 'cifar100c', 'imagenet', 'imagenet3dcc'],
                        help='Dataset to use')
    parser.add_argument('--threat-model', type=str, default='Linf',
                        choices=['Linf', 'L2', 'corruptions'],
                        help='Threat model for robust training')
    parser.add_argument('--n-examples', type=int, default=5,
                        help='Number of examples to process')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    return parser.parse_args()

def main(model, dataset, threat_model, n_examples, output):
    print(f"Model: {model}, Dataset: {dataset}, Threat model: {threat_model}, Number of examples: {n_examples}, Output path: {output}")

    # Notes on datasets: ImageNet must be downloaded manually. 
    # See https://github.com/RobustBench/robustbench?tab=readme-ov-file#model-zoo
    if dataset == 'cifar10':
        x_test, y_test = load_cifar10(n_examples=n_examples)
    else:
        print(f"unsupported dataset {dataset}")
        return

    model = load_model(model_name=model, dataset=dataset, threat_model=threat_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using device", device)
    model.to(device)
    x_test = x_test.to(device)
    model.eval() # no grad descent

    # Run inference
    predictions = []
    with torch.no_grad():
        for x in x_test:
            x = x.unsqueeze(0)  # Add batch dimension (x.shape 3,32,32 -> 1,3,32,32)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            predictions.append(pred.item())

    # Flatten x.shape n,3,32,32 -> n,3072
    x_test = x_test.reshape(n_examples, -1)

    # save csv
    df = pd.DataFrame({
        'pred': predictions,
        **{f'f{i}': x_test[:, i].cpu().numpy() for i in range(x_test.shape[1])}
    })
    df.to_csv(output, index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args.model, args.dataset, args.threat_model, args.n_examples, args.output)

