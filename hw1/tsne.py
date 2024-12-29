import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


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


def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)


def apply_tsne(features):
    tsne = TSNE(n_components=2, random_state=42)
    return tsne.fit_transform(features)


def visualize_tsne(tsne_results, labels, epoch):
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="viridis", s=5)
    plt.colorbar()
    plt.title(f"t-SNE Visualization of Learned Features - Epoch {epoch}")
    plt.savefig(f"t-SNE-{epoch}")
    plt.show()


batch_size = 32

train_dataset = OfficeDataset("../hw1_data/p1_data/office/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

resnet_model = models.resnet50(weights=None)
pre_trained_model = SelfSupervisedLearner.load_from_checkpoint(
    checkpoint_path="/kaggle/input/pretrained_model_ssl/pytorch/default/1/pretrain_model_SSL.ckpt",
    network=resnet_model,
    image_size=128,
    hidden_layer="avgpool",
    use_momentum=False,
    projection_size=256,
    projection_hidden_size=4096,
    moving_average_decay=0.99,
)

first_resnet_model = pre_trained_model.learner.online_encoder.net

resnet_model = models.resnet50(weights=None)
num_classes = 65

last_model = ImageClassifier.load_from_checkpoint(
    checkpoint_path="/kaggle/input/c-best-checkpoint/pytorch/default/1/best-checkpoint.ckpt",
    pretrained_model=resnet_model,
    num_classes=num_classes,
)

last_resnet_model = last_model.model

features_first_epoch, labels = extract_features(first_resnet_model, train_loader)
tsne_first_epoch = apply_tsne(features_first_epoch)
visualize_tsne(tsne_first_epoch, labels, epoch=1)

features_last_epoch, labels = extract_features(last_resnet_model, train_loader)
tsne_last_epoch = apply_tsne(features_last_epoch)
visualize_tsne(tsne_last_epoch, labels, epoch=200)
