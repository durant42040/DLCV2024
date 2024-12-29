import os

import pytorch_lightning as pl
import torch
import torchvision
from tqdm import tqdm

from dataset import DDPMDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, random_split

from UNet import UNet


class DDPM(pl.LightningModule):
    def __init__(self, model, cond_scale=0.5, T=200):
        super().__init__()
        self.model = model
        self.cond_scale = cond_scale
        self.T = T

        alpha = torch.linspace(0.999, 0.99, T)
        self.register_buffer("alpha", alpha)

        alpha_bar = torch.cumprod(self.alpha, dim=-1).reshape(-1, 1)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x, t, c, d):
        return self.model(x, t, c, d)

    def denoise_step(self, x, t, c, d):
        if c is not None:
            return (1 + self.cond_scale) * self.model(
                x, t, c, d
            ) - self.cond_scale * self.model(x, t, None, d)
        else:
            return self.model(x, t, None, d)

    def diffusion_step(self, x, t, noise):
        batch_size = x.size(0)
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1, 1).to(x.device)

        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

    def training_step(self, batch, batch_idx):
        x0, c, d = batch
        t = torch.randint(0, self.T, (x0.size(0),), device=self.device)
        noise = torch.randn_like(x0)

        x_t = self.diffusion_step(x0, t, noise)

        cond_input = c if torch.rand(()) > 0.1 else None

        pred_noise = self.model(x_t, t, cond_input, d)
        loss = nn.functional.mse_loss(pred_noise, noise)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x0, c, d = batch
        t = torch.randint(0, self.T, (x0.size(0),), device=self.device)
        noise = torch.randn_like(x0)

        x_t = self.diffusion_step(x0, t, noise)

        use_cond = torch.rand(()) > 0.1
        cond_input = c if use_cond else None

        pred_noise = self.model(x_t, t, cond_input, d)
        loss = nn.functional.mse_loss(pred_noise, noise)

        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)

    def sample(self, shape, d, c=None):
        x = torch.randn(shape).to(self.device)

        for t in tqdm(reversed(range(self.T)), desc="Sampling Steps"):
            t_tensor = torch.full((x.shape[0],), t, dtype=torch.long).to(self.device)

            eps = self.denoise_step(x, t_tensor, c, d)

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)

            x = (1 / torch.sqrt(self.alpha[t])) * (
                x - (1 - self.alpha[t]) * eps / torch.sqrt(1 - self.alpha_bar[t])
            ) + torch.sqrt(1 - self.alpha[t]) * z

        return x


if __name__ == "__main__":
    cond_scale = 5
    T = 500
    torch.manual_seed(42)
    os.environ["WANDB_API_KEY"] = "cea091384aba595fb3f51ee372ed755a0ac3ba5d"

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    unet = UNet(
        T=T,
        num_labels=10,
        num_datasets=2,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        num_res_blocks=2,
        dropout=0.1,
    )
    ddpm = DDPM(unet, cond_scale=cond_scale, T=T)

    dataset = DDPMDataset(
        mnistm_path="hw2_data/digits/mnistm",
        svhn_path="hw2_data/digits/svhn",
        transform=transform,
    )

    dataset_size = len(dataset)
    train_size = int(0.99 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # print(ddpm)
    wandb_logger = WandbLogger(project="DLCV-hw2-p1", log_model=True)
    wandb_logger.log_hyperparams({"cond_scale": cond_scale, "T": T})

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.5f}",
        save_top_k=1,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gpus=1,
    )

    trainer.fit(ddpm, train_loader, val_loader)
