import h5py
import numpy as np
import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data/nuplan_bev/mini/nuplan_mini_val_10hz_gt_masks_bev.h5')
parser.add_argument('--output-dir', default='data/nuplan_bev_400/mini/val')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with h5py.File(args.input, 'r') as f:
    keys = list(f.keys())
    for key in tqdm(keys, desc="Reading HDF5 groups"):
        grp = f[key]
        tokens = list(grp.keys())
        for token in tqdm(tokens, desc=f"Processing group {key}"):
            np.save(os.path.join(args.output_dir, f"{token}.npy"), grp[token][:])

print("Done")
