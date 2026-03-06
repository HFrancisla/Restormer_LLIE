## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
import time
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(
    description="Image Low-Light Enhancement using Restormer"
)

parser.add_argument(
    "--input_dir",
    default="./datasets/LOL-v2/Real_captured/Test/Low/",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir", default="./results/LOL-v2/", type=str, help="Directory for results"
)
parser.add_argument(
    "--weights",
    default="./pretrained_models/lowlight.pth",
    type=str,
    help="Path to weights",
)
parser.add_argument(
    "--opt",
    type=str,
    default="LLIE/Options/LowLight_Restormer_FFT_128_2_60k_1.yml",
    help="Path to option YAML file.",
)


args = parser.parse_args()

####### Load yaml #######
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(args.opt, mode="r", encoding="utf-8"), Loader=Loader)

s = x["network_g"].pop("type")
##########################

model_restoration = Restormer(**x["network_g"])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()
# Automatically get the padding factor from the yaml configuration's val window_size
window_size = x.get("val", {}).get("window_size", 8)
factor = window_size
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

inp_dir = args.input_dir
files = natsorted(
    glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.jpg"))
)

print(f"\n===> Processing {len(files)} images...")
total_inference_time = 0
image_count = 0

with torch.no_grad():
    for file_ in tqdm(files):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_)) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), "reflect")

        # Start timing for inference
        start_time = time.time()
        restored = model_restoration(input_)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end_time = time.time()

        # Record inference time
        inference_time = end_time - start_time
        total_inference_time += inference_time
        image_count += 1

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = (
            torch.clamp(restored, 0, 1)
            .cpu()
            .detach()
            .permute(0, 2, 3, 1)
            .squeeze(0)
            .numpy()
        )

        utils.save_img(
            (
                os.path.join(
                    result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + ".png"
                )
            ),
            img_as_ubyte(restored),
        )

# Display timing statistics
print("\n" + "=" * 60)
print("INFERENCE TIMING STATISTICS")
print("=" * 60)
print(f"Total images processed: {image_count}")
print(
    f"Total inference time: {total_inference_time:.4f} seconds ({total_inference_time / 60:.2f} minutes)"
)
if image_count > 0:
    avg_time = total_inference_time / image_count
    print(f"Average time per image: {avg_time:.4f} seconds ({avg_time * 1000:.2f} ms)")
    print(f"Processing speed: {1 / avg_time:.2f} images/second")
print("=" * 60)
