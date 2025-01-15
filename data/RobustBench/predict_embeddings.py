import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import time
from datetime import datetime
from robustbench.data import load_cifar10
from robustbench.utils import load_model

from predict import make_argparser, load_dataset, log, classify

def extend_argparser(parser):
    parser.add_argument('--emb_model', type=str, default='small',
                        choices=['small', 'base', 'large', 'giant'],
                        help='Name of the embedding model to use')
    parser.add_argument('--force_resize', action='store_true',
                        help='Force resizing of the embeddings')
    return parser

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

def process_img(img, force_resize):
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
    predictions = []
    last_update = time.time()

    with torch.no_grad():
        for i, x in enumerate(data):
            x = x.unsqueeze(0)  # Add batch dimension (x.shape 3,32,32 -> 1,3,32,32)
            predictions.append(model(x))

            if time.time() - last_update > 1:
                log(f"dino: {i}")
                last_update = time.time()

    return predictions

if __name__ == '__main__':
    args = extend_argparser(make_argparser()).parse_args()
    log(f"Model: {args.model}, Dataset: {args.dataset}, Threat model: {args.threat_model}, Number of examples: {args.n_examples}, Output path: {args.output}")

    model = load_model(model_name=args.model, dataset=args.dataset, threat_model=args.threat_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval() # no grad descent

    x_test, _ = load_dataset(args.dataset, args.n_examples)
    x_test = x_test.to(device)
    log(f"{args.model} model and {len(x_test)} samples from {args.dataset} loaded on device {device}")

    # classifier predictions
    predictions = [i.item() for i in classify(model, x_test)]
    
    # Free up GPU memory
    if torch.cuda.is_available():
        model.cpu()
        torch.cuda.empty_cache()
    del model

    dino = load_dino(args.emb_model, device)
    log(f"dino {args.model} model loaded on device {device}")

    # embeddings
    processed_imgs = [process_img(img, args.force_resize) for img in x_test]
    embeddings = embed(dino, processed_imgs)

    # save csv
    embeddings_array = torch.cat(embeddings, dim=0).cpu().numpy()
    df = pd.DataFrame(
        np.column_stack([predictions, embeddings_array]),
        columns=['pred'] + [f'f{i}' for i in range(embeddings_array.shape[1])]
    )
    df.to_csv(args.output, index=False)
    log("completed")

