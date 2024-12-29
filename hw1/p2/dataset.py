import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

mask_table = {3: 0, 6: 1, 5: 2, 2: 3, 1: 4, 7: 5, 0: 6}


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.img_list = [f for f in os.listdir(img_dir) if f.endswith("_sat.jpg")]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        seed = random.randint(0, 2**32)
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        mask_path = os.path.join(
            self.img_dir,
            self.img_list[idx].replace("_sat", "_mask").replace(".jpg", ".png"),
        )
        print("img_path", img_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        mask = np.array(mask)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        mask = np.vectorize(lambda x: mask_table.get(x, x))(mask)

        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        mask.squeeze_(0)

        return image, mask


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomCrop(512, padding=30, padding_mode='reflect'),
            # transforms.RandomResizedCrop(512),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    #
    # dataset = SegmentationDataset("../hw1_data/p2_data/train", transform=transform)
    #
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    #
    # for i, (image, mask) in enumerate(dataloader):
    #     print(image.shape)
    #     print(mask.shape)
    #     print(np.unique(mask))
    #     break

    seed = random.randint(0, 2**32)
    img_path = "../hw1_data/p2_data/train/0020_sat.jpg"
    mask_path = "../hw1_data/p2_data/train/0020_mask.png"

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    mask = np.array(mask)
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    mask = np.vectorize(lambda x: mask_table.get(x, x))(mask)

    if transform:
        torch.manual_seed(seed)
        image = transform(image)
        torch.manual_seed(seed)
        mask = transform(mask)

    mask.squeeze_(0)

    print(mask.shape, image.shape)
    print(np.unique(mask))
    # mask = torch.tensor(mask, dtype=torch.long)
