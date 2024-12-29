import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

from p2.dataset import SegmentationDataset

image_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)


class FCN32s(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = vgg16.features

        self.fc_conv = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, kernel_size=1),
        )

        self.upsample32x = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False
        )

        self.iou = MulticlassJaccardIndex(num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc_conv(x)
        x = self.upsample32x(x)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


num_classes = 7

if __name__ == "__main__":
    train_dataset = SegmentationDataset(
        "../hw1_data/p2_data/train", transform=image_transform
    )
    val_dataset = SegmentationDataset(
        "../hw1_data/p2_data/validation", transform=image_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    trainer = pl.Trainer(max_epochs=10)
    model = FCN32s(num_classes=num_classes)

    trainer.fit(model, train_loader, val_loader)
