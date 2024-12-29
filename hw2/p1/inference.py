import argparse

import torch
import random
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
from torchvision.transforms.functional import to_pil_image

from ddpm import DDPM
from UNet import UNet

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def generate_and_save_images(model, device, save_dir, num_samples_per_digit=50):
    model.eval()
    model.to(device)

    num_digits = 10
    num_datasets = 2
    total_samples = num_samples_per_digit * num_digits * num_datasets

    c = torch.repeat_interleave(
        torch.arange(num_digits), num_samples_per_digit * num_datasets
    ).to(device)
    d = (
        torch.tensor([0] * num_samples_per_digit + [1] * num_samples_per_digit)
        .repeat(num_digits)
        .to(device)
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            sampled_images = model.sample(shape=(total_samples, 3, 32, 32), c=c, d=d)
            sampled_images = (sampled_images.cpu() + 1) / 2

    mnistm_dir = os.path.join(save_dir, "mnistm")
    svhn_dir = os.path.join(save_dir, "svhn")
    os.makedirs(mnistm_dir, exist_ok=True)
    os.makedirs(svhn_dir, exist_ok=True)

    def save_image(idx):
        img = sampled_images[idx]
        digit = c[idx].item()
        dataset = d[idx].item()
        sample_num = (idx % num_samples_per_digit) + 1
        sample_num_str = f"{sample_num:03d}"
        filename = f"{digit}_{sample_num_str}.png"
        img = to_pil_image(img)
        if dataset == 0:
            save_path = os.path.join(mnistm_dir, filename)
        else:
            save_path = os.path.join(svhn_dir, filename)
        img.save(save_path)

    with ThreadPoolExecutor() as executor:
        executor.map(save_image, range(total_samples))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM Image Generation")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = 500
    cond_scale = 10

    unet = UNet(
        T=T,
        num_labels=10,
        num_datasets=2,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        dropout=0.1,
    )
    ddpm = DDPM(unet, cond_scale=cond_scale, T=T)

    checkpoint_path = "p1_model.ckpt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(state_dict["state_dict"])

    ddpm.to(device)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    num_samples_per_digit = 50

    start_time = time.time()

    generate_and_save_images(
        ddpm, device, output_dir, num_samples_per_digit=num_samples_per_digit
    )

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Image generation and saving completed in {elapsed_time:.2f} seconds.")
