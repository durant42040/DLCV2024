import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

mask_table = {3: 0, 6: 1, 5: 2, 2: 3, 1: 4, 7: 5, 0: 6}

color_table = {
    0: [0, 255, 255],
    1: [255, 255, 0],
    2: [255, 0, 255],
    3: [0, 255, 0],
    4: [0, 0, 255],
    5: [255, 255, 255],
    6: [0, 0, 0],
}


def mean_iou(pred, labels):
    mean_iou = 0
    pred = pred.flatten()
    labels = labels.flatten()
    for i in range(6):
        tp_fp = torch.sum(pred == i).item()
        tp_fn = torch.sum(labels == i).item()
        tp = torch.sum((pred == i) & (labels == i)).item()
        if tp_fp + tp_fn - tp == 0:
            iou = 0
        else:
            iou = tp / (tp_fp + tp_fn - tp)
        # iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    #     print(f'class #{i} : {iou:.5f}')
    # print(f'\nmean_iou: {mean_iou:.5f}\n')

    return mean_iou


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
        # mask_path = os.path.join(
        #     self.img_dir,
        #     self.img_list[idx].replace("_sat", "_mask").replace(".jpg", ".png"),
        # )

        image = Image.open(img_path).convert("RGB")
        # mask = Image.open(mask_path).convert("RGB")

        # mask = np.array(mask)
        # mask = (mask >= 128).astype(int)
        # mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        # mask = np.vectorize(lambda x: mask_table.get(x, x))(mask)

        if self.transform:
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            # mask = self.transform(mask)

        # mask.squeeze_(0)

        return image, self.img_list[idx]


class DeepLab(nn.Module):
    def __init__(self, num_classes, mean_iou):
        super(DeepLab, self).__init__()

        # Load pretrained DeepLabV3 with ResNet50 backbone
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "deeplabv3_resnet50",
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
        )

        # Replace classifier with the one for the specified number of classes
        self.model.classifier = DeepLabHead(2048, num_classes)

        # IOU function (mean IoU)
        self.iou = mean_iou

    def forward(self, x):
        # Forward pass through DeepLabV3
        x = self.model(x)["out"]
        return x

    def training_step(self, x, y):
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        iou = self.iou(preds, y)

        return loss, iou

    def validation_step(self, x, y):
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        iou = self.iou(preds, y)

        return loss, iou

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=150)

        return optimizer, scheduler


def inference(model, dataloader, device):
    model.eval()

    with torch.no_grad():
        for i, (images, name) in enumerate(dataloader):
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for j, pred in enumerate(preds):
                color_mask = torch.zeros(
                    (pred.shape[0], pred.shape[1], 3), dtype=torch.uint8, device=device
                )

                for label, color in color_table.items():
                    color_tensor = torch.tensor(color, dtype=torch.uint8, device=device)
                    color_mask[pred == label] = color_tensor

                color_mask = color_mask.cpu().numpy()
                output_image = Image.fromarray(color_mask)
                output_image.save(
                    os.path.join(output_dir, name[j].replace("_sat.jpg", "_mask.png"))
                )
                print(f"Saved {name[j].replace('_sat.jpg', '_mask.png')}")


image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir", type=str, help="path to the folder containing images"
    )
    parser.add_argument("output_dir", type=str, help="path to the output folder")

    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    num_classes = 7

    dataset = SegmentationDataset(input_dir, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("best-deeplab-model.ckpt", map_location=device)
    state_dict = checkpoint["state_dict"]

    model = DeepLab(num_classes, mean_iou)
    model.load_state_dict(state_dict)
    model.to(device)

    inference(model, dataloader, device)
