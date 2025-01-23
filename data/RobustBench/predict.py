import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import time
from datetime import datetime
from robustbench.data import load_cifar10, load_cifar100, load_cifar10c, load_cifar100c, load_imagenet, load_imagenet3dcc
from robustbench.utils import load_model

def log(s):
    timestamp = datetime.utcfromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {s}", flush=True)

def fatal(s):
    log(s)
    exit(1)

def make_argparser():
    parser = argparse.ArgumentParser(description='Run inference with RobustBench models')
    parser.add_argument('--model', type=str, default='Standard',
                        help='Name of the model to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar10c', 'cifar100', 'cifar100c', 'imagenet', 'imagenet3dcc'],
                        help='Dataset to use')
    parser.add_argument('--threat-model', '--threat_model', type=str, default='Linf',
                        choices=['Linf', 'L2', 'corruptions', 'corruptions_3d'],
                        help='Threat model for robust training')
    parser.add_argument('--n-examples', '--n_examples', type=int, default=10000,
                        help='Number of examples to process')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--emb-model', '--emb_model', type=str, default='none',
                        choices=['none', 'small', 'base', 'large', 'giant'],
                        help='Name of the embedding model to use')
    parser.add_argument('--force-resize', '--force_resize', action='store_true',
                        help='always resize images to have smaller dimension of 224px for the embedding model')
    parser.add_argument('--corruption', type=str, default=None,
                        choices=["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur", 
                                 "brightness", "fog", "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast", 
                                 "jpeg_compression", "elastic_transform"],
                        help='which corruption (cifar10c, cifar100c only)')
    parser.add_argument('--corruption3d', type=str, default=None,
                        choices=['near_focus', 'far_focus', 'fog_3d', 'flash', 'color_quant', 'low_light', 
                                 'xy_motion_blur', 'z_motion_blur', 'iso_noise', 'bit_error', 'h265_abr', 'h265_crf'],
                        help='which 3d corruption (imagenet3dcc only)')
    parser.add_argument('--severity', type=int, default=5,
                        help='severity of corruption (corruption datasets only)')
    return parser

def basename(dataset_name):
    if 'cifar100' in dataset_name:
        return 'cifar100'
    elif 'cifar10' in dataset_name:
        return 'cifar10'
    elif 'imagenet' in dataset_name:
        return 'imagenet'
    else:
        raise ValueError(f"unknown base dataset for {name}")

def load_dataset(name, n_examples, corruption=None, corruption3d=None, severity=5):
    """Notes on datasets: ImageNet must be downloaded manually. 
    ImageNet instructions: (from https://github.com/RobustBench/robustbench?tab=readme-ov-file#model-zoo)
    Download and untar the validation set in ./data/val (create if needed)
    In that directory, run the following script:
    https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

    ImageNet-3DCC instructions:
    Download the dataset using the provided script:
    https://github.com/EPFL-VILAB/3DCommonCorruptions/blob/main/tools/download_imagenet_3dcc.sh
    Place the ImageNet-3DCC folder in ./data/
    """

    if name == 'cifar10':
        x_test, y_test = load_cifar10(n_examples=n_examples)
    elif name == 'cifar100':
        x_test, y_test = load_cifar100(n_examples=n_examples)
    elif name == 'imagenet':
        x_test, y_test = load_imagenet(n_examples=n_examples)
    elif name == 'cifar10c':
        if not corruption: fatal("cifar10c requires specifying a corruption")
        x_test, y_test = load_cifar10c(n_examples=n_examples, corruptions=[corruption], severity=severity)
    elif name == 'cifar100c':
        if not corruption: fatal("cifar100c requires specifying a corruption")
        x_test, y_test = load_cifar100c(n_examples=n_examples, corruptions=[corruption], severity=severity)
    elif name == 'imagenet3dcc':
        if not corruption3d: fatal("imagenet3dcc requires specifying a 3d corruption")
        x_test, y_test = load_imagenet3dcc(n_examples=n_examples, corruptions=[corruption3d], severity=severity)
    else:
        raise ValueError(f"unsupported dataset {name}")
    return x_test, y_test

def load_dino(backbone, device):
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[backbone]
    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=f"dinov2_{backbone_arch}")
    model.to(device)
    model.eval()
    return model

def process_img(img, force_resize, device):
    img = transforms.ToPILImage()(img)
    width, height = img.size
    
    # Scale image to have smaller dimension of max. 224px while maintaining aspect ratio
    if force_resize or (width > 224 and height > 224):
        scale = 224.0 / min(width, height)
        resize_width = int(width * scale)
        resize_height = int(height * scale)
    else:
        resize_width = width
        resize_height = height
    
    # Ensure dimensions are divisible by 14
    crop_width = resize_width - (resize_width % 14)
    crop_height = resize_height - (resize_height % 14)
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.CenterCrop((crop_height, crop_width)),
        transforms.ToTensor(),
    ])
    return transform(img).to(device)

def embed(model, data):
    embeddings = []
    last_update = time.time()

    with torch.no_grad():
        for i, x in enumerate(data):
            x = x.unsqueeze(0)
            embeddings.append(model(x))

            if time.time() - last_update > 1:
                log(f"dino: {i}")
                last_update = time.time()

    return embeddings

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
                log(f"classifier: {i}")
                last_update = time.time()

    return predictions

def format(predictions, embeddings, x_test, y_test):
    if embeddings:
        embeddings_array = torch.cat(embeddings, dim=0).cpu().numpy()
        return pd.DataFrame(
            np.column_stack([predictions, y_test, embeddings_array]),
            columns=['pred', 'label'] + [f'e{i}' for i in range(embeddings_array.shape[1])]
        )

    else:
        x_test = x_test.reshape(len(x_test), -1)
        return pd.DataFrame({
            'pred': predictions,
            'label': y_test,
            **{f'f{i}': x_test[:, i].cpu().numpy() for i in range(x_test.shape[1])}
        })


if __name__ == '__main__':
    args = make_argparser().parse_args()
    log(f"Model: {args.model}, Dataset: {args.dataset}, Threat model: {args.threat_model}, Number of examples: {args.n_examples}, Output path: {args.output}")

    model = load_model(model_name=args.model, dataset=basename(args.dataset), threat_model=args.threat_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # no grad descent
    log(f"{args.model} model loaded on device {device}")

    x_test, y_test = load_dataset(args.dataset, args.n_examples, args.corruption, args.corruption3d, args.severity)
    x_test = x_test.to(device)
    log(f"{len(x_test)} samples from {args.dataset} loaded on device {device}")

    predictions = [i.item() for i in classify(model, x_test)]
    log(f"finished computing classifier predictions")
    
    embeddings = []
    if args.emb_model != 'none':
        # free GPU memory for dino
        if device == 'cuda':
            model.cpu()
            torch.cuda.empty_cache()
        del model

        dino = load_dino(args.emb_model, device)
        log(f"dino {args.emb_model} model loaded on device {device}")

        processed_imgs = [process_img(img, args.force_resize, device) for img in x_test]
        embeddings = embed(dino, processed_imgs)
        log(f"finished computing embeddings")

    log("saving csv...")
    format(predictions, embeddings, x_test, y_test).to_csv(args.output, index=False)

    log("completed")

