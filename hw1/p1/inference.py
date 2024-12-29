import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


class OfficeDataset(Dataset):
    def __init__(self, image_dir, transform=None, image_data=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_data = image_data

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_data["filename"][idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        id = self.image_data["id"][idx]
        filename = self.image_data["filename"][idx]

        return image, id, filename


class ImageClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(ImageClassifier, self).__init__()
        self.model = pretrained_model

        # Freeze pretrained model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Add fully connected layer, batch normalization, and dropout
        self.fc = nn.Linear(1000, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Forward pass through pretrained model
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)
        return x

    def training_step(self, x, y):
        # Training step: forward pass, compute loss, accuracy
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        return loss, acc

    def validation_step(self, x, y):
        # Validation step: forward pass, compute loss, accuracy
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        return loss, acc


num_classes = 65


def inference(model, dataloader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for images, ids, filenames in dataloader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            preds = preds.cpu().numpy()

            for id, pred, filename in zip(ids, preds, filenames):
                results.append({"id": id.item(), "filename": filename, "label": pred})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="path to the images csv file")
    parser.add_argument(
        "input_folder", type=str, help="path to the folder containing images"
    )
    parser.add_argument("output_csv", type=str, help="path of output .csv file")

    args = parser.parse_args()

    input_csv = args.input_csv
    input_folder = args.input_folder
    output_csv = args.output_csv

    image_data = pd.read_csv(input_csv)

    dataset = OfficeDataset(input_folder, transform=transform, image_data=image_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    resnet_model = models.resnet50(weights=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("best-resnet-model.ckpt", map_location=device)
    state_dict = checkpoint["state_dict"]

    model = ImageClassifier(resnet_model, num_classes)
    model.load_state_dict(state_dict)
    model.to(device)

    results = inference(model, dataloader, device)

    # read image data and save results
    image_data = pd.DataFrame(results)
    image_data = image_data.sort_values(by="id")

    image_data.to_csv(output_csv, index=False)
