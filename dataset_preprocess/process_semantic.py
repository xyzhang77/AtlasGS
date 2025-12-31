import argparse
import os
import torch
import gzip
import numpy as np
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='Process semantic segmentation data')
    parser.add_argument('--input', type=str, default=None, help='Path to the input data')
    return parser.parse_args()

def main():
    args = get_parser()

    semantic_folder = os.path.join(args.input, "panoptic")
    
    for file in tqdm(os.listdir(semantic_folder)):
        if file.endswith(".ptz"):
            with gzip.open(os.path.join(semantic_folder, file), 'rb') as f:
                data = torch.load(f)
            
            unlabeled_mask = data["probabilities"][..., 0] == 1
            probabilities = data["probabilities"][..., 1:]
            probabilities[..., 3] += data["probabilities"][..., 0]
            confidence = data['confidences']
            confidence[unlabeled_mask] = 0.1
            np.savez_compressed(os.path.join(semantic_folder, file.replace(".ptz", ".npz")), probabilities=probabilities.cpu().numpy(), confidence=confidence.cpu().numpy())

if __name__ == "__main__":
    main()