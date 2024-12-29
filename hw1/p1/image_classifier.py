import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# from p1.pretrain import SelfSupervisedLearner
import torchmetrics
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomCrop(128, padding=4, padding_mode="reflect"),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
    ]
)

num_classes = 65


class OfficeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(self.image_filenames[idx].split("_")[0])

        return image, label


class ImageClassifier(pl.LightningModule):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.model = pretrained_model

        for param in self.model.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(1000, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        self.dropout = nn.Dropout(0.5)

        self.train_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    train_dataset = OfficeDataset("hw1_data/p1_data/office/train", transform=transform)
    val_dataset = OfficeDataset("hw1_data/p1_data/office/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    resnet_model = models.resnet50(weights=None)  # No pretrained weights

    pre_trained_model = SelfSupervisedLearner.load_from_checkpoint(
        checkpoint_path="pretrain_model_SSL.ckpt",
        network=resnet_model,
        image_size=128,
        hidden_layer="avgpool",
        use_momentum=False,
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99,
    )

    resnet_model = pre_trained_model.learner.online_encoder.net
    #     resnet_model.load_state_dict(torch.load("/kaggle/input/pretrained_model_sl/pytorch/default/1/pretrain_model_SL.pt"))

    model = ImageClassifier(resnet_model, num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
    )

    trainer = pl.Trainer(max_epochs=200, callbacks=[checkpoint_callback])

    trainer.fit(model, train_loader, val_loader)
