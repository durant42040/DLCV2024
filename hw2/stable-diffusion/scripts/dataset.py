import json
import os
import random
from random import randint

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

template_strings = [
    "a photo of a {}.",
    "a rendering of a {}.",
    "a cropped photo of the {}.",
    "the photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a good photo of a {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "a photo of the large {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
]


class TextualInversionDataset(Dataset):
    def __init__(self, data_root, image_size=(512, 512)):
        super().__init__()
        self.data_root = data_root

        with open(os.path.join(data_root, "input.json"), "r") as f:
            self.metadata = json.load(f)

        self.img_paths = []
        self.prompts = []
        for key, value in self.metadata.items():
            category_path = os.path.join(data_root, key)
            if os.path.isdir(category_path):
                category_images = [
                    os.path.join(category_path, fname)
                    for fname in os.listdir(category_path)
                    if fname.endswith(".jpg") or fname.endswith(".png")
                ]
                self.img_paths.extend(category_images)
                for _ in category_images:
                    prompt = template_strings[
                        randint(0, len(template_strings) - 1)
                    ].replace("{}", value["token_name"])
                    self.prompts.append(prompt)

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        prompt = self.prompts[idx]
        return image, prompt


def main():
    seed = 42
    random.seed(seed)
    data_root = "/Users/electron/Code/DLCV2024/hw2/hw2_data/textual_inversion"
    batch_size = 4

    dataset = TextualInversionDataset(data_root=data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch in dataloader:
        images, prompts = batch
        print(f"Batch size: {len(images)}")
        print(f"Prompts: {prompts}")
        break


if __name__ == "__main__":
    main()
