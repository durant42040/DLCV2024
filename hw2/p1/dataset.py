import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class DDPMDataset(Dataset):
    def __init__(self, mnistm_path, svhn_path, transform=None):
        self.mnistm_data = pd.read_csv(os.path.join(mnistm_path, "train.csv"))
        self.mnistm_img_dir = os.path.join(mnistm_path, "data")

        self.svhn_data = pd.read_csv(os.path.join(svhn_path, "train.csv"))
        self.svhn_img_dir = os.path.join(svhn_path, "data")

        self.transform = transform

    def __len__(self):
        return len(self.mnistm_data) + len(self.svhn_data)

    def __getitem__(self, idx):
        if idx < len(self.mnistm_data):
            img_name = self.mnistm_data.iloc[idx, 0]
            img_path = f"{self.mnistm_img_dir}/{img_name}"
            label = self.mnistm_data.iloc[idx, 1]
            d = torch.tensor(0, dtype=torch.long)
        else:
            idx -= len(self.mnistm_data)
            img_name = self.svhn_data.iloc[idx, 0]
            img_path = f"{self.svhn_img_dir}/{img_name}"
            label = self.svhn_data.iloc[idx, 1]
            d = torch.tensor(1, dtype=torch.long)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, d


if __name__ == "__main__":
    mnistm_dataset = DDPMDataset(
        mnistm_path="hw2_data/digits/mnistm",
        svhn_path="hw2_data/digits/svhn",
        transform=None,
    )
    train_loader = DataLoader(mnistm_dataset, batch_size=8, shuffle=True)

    for batch in train_loader:
        images, labels, d = batch
        for i in range(len(images)):
            print(labels[i].item(), d[i].item())

        quit()
