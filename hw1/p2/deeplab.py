import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead, DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights)

mask_table = {3: 0, 6: 1, 5: 2, 2: 3, 1: 4, 7: 5, 0: 6}

image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomApply(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)
                ),
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
                transforms.RandomCrop(512, padding=10, padding_mode="reflect"),
                transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            ],
            p=0.5,
        ),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


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
        # mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


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


class DeepLab(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "deeplabv3_resnet50",
            weights=DeepLabV3_ResNet50_Weights.DEFAULT,
        )
        self.model.classifier = DeepLabHead(2048, num_classes)

        # self.jaccard = MulticlassJaccardIndex(num_classes=num_classes)
        self.iou = mean_iou

    def forward(self, x):
        x = self.model(x)["out"]
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        iou = self.iou(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_iou", iou, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        iou = self.iou(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_iou", iou, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
        return [optimizer], [scheduler]


num_classes = 7

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train_dataset = SegmentationDataset(
        "../hw1_data/p2_data/train", transform=image_transform
    )
    val_dataset = SegmentationDataset(
        "../hw1_data/p2_data/validation", transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=7)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_iou",
        mode="max",
        filename="{epoch}-{val_iou:.3f}-{val_loss:.3f}",
        save_top_k=1,
        verbose=True,
    )

    epoch_callback = ModelCheckpoint(
        filename="{epoch}-{val_iou:.3f}-{val_loss:.3f}",
        verbose=True,
        every_n_epochs=30,
        save_top_k=-1,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[checkpoint_callback, epoch_callback],
    )
    model = DeepLab(num_classes=num_classes)

    trainer.fit(model, train_loader, val_loader)
