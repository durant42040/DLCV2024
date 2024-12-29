import json
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, img_path, json_path, transform=None):
        self.image_path = img_path
        self.json_path = json_path
        self.transform = transform

        with open(self.json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data["annotations"])

    def __getitem__(self, idx):
        caption = self.data["annotations"][idx]["caption"]
        image_id = self.data["annotations"][idx]["image_id"]

        image = Image.open(
            os.path.join(self.image_path, f"{image_id:012}.jpg")
        ).convert("RGB")

        # caption = "<|endoftext|>Write a description for the image in one sentence.<|endoftext|>" + caption + "<|endoftext|>"
        caption = "<|endoftext|>" + caption + "<|endoftext|>"

        if self.transform:
            image = self.transform(image)

        return image, caption


class InferenceDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.image_path = img_path
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, f"{idx:012}.jpg")).convert(
            "RGB"
        )

        if self.transform:
            image = self.transform(image)

        return idx, image


def main():
    train_dataset = ImageDataset(
        "hw3_data/p2_data/images/train",
        "hw3_data/p2_data/train.json",
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )
    # train_dataset = InferenceDataset(
    #     "hw3_data/p2_data/images/train",
    #     "hw3_data/p2_data/train.json",
    #     transform=transforms.Compose(
    #         [transforms.Resize((224, 224)), transforms.ToTensor()]
    #     ),
    # )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    print(len(train_loader))
    for batch_idx, images in enumerate(train_loader):
        print(len(train_loader))
        print(f"Batch {batch_idx}")
        break


if __name__ == "__main__":
    main()
