import os

import pytorch_lightning as pl
import torch
from dataset import TextualInversionDataset
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

import wandb
from ldm.util import instantiate_from_config


class TextualInversion(pl.LightningModule):
    def __init__(self, config, learning_rate, ckpt, new_tokens):
        super().__init__()

        self.model = self.load_model_from_config(config, ckpt)
        self.learning_rate = learning_rate
        self.new_tokens = new_tokens

        for param in self.model.parameters():
            param.requires_grad = False

        self.init_new_tokens()
        # self.check_trainable_parameters()

    def check_trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"Trainable parameter: {name}")
        if not any(param.requires_grad for param in self.parameters()):
            print("No trainable parameters found.")

    def load_model_from_config(self, config, ckpt):
        model = instantiate_from_config(config.model)
        sd = torch.load(ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)

        return model

    def init_new_tokens(self):
        tokenizer = self.model.cond_stage_model.tokenizer

        num_added_tokens = tokenizer.add_tokens(self.new_tokens)
        if num_added_tokens != len(self.new_tokens):
            raise ValueError("Some placeholder tokens were already in the tokenizer.")

        transformer = self.model.cond_stage_model.transformer
        transformer.resize_token_embeddings(len(tokenizer))

        embedding_layer = transformer.get_input_embeddings()
        embedding_layer.weight.requires_grad = True

        new_token_ids = tokenizer.convert_tokens_to_ids(self.new_tokens)

        for token_id in new_token_ids:
            embedding_layer.weight.data[token_id] = torch.randn(
                embedding_layer.embedding_dim
            )
            embedding_layer.weight.data[token_id] /= embedding_layer.embedding_dim**0.5

    def forward(self, images, prompts):
        conditioning = self.model.get_learned_conditioning(prompts)
        t = torch.randint(
            0, self.model.num_timesteps, (images.shape[0],), device=images.device
        ).long()
        loss, _ = self.model.p_losses(images, conditioning, t)
        return loss

    def training_step(self, batch, batch_idx):
        images, prompts = batch
        loss = self.forward(images, prompts)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        transformer = self.model.cond_stage_model.transformer
        embedding_layer = transformer.get_input_embeddings()
        optimizer = torch.optim.Adam(embedding_layer.parameters(), lr=self.learning_rate)
        return optimizer


def main():
    os.environ["WANDB_API_KEY"] = "cea091384aba595fb3f51ee372ed755a0ac3ba5d"
    data_root = "/Users/electron/Code/DLCV2024/hw2/hw2_data/textual_inversion"
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    ckpt = "ldm/models/stable-diffusion-v1/model.ckpt"

    learning_rate = 1e-4
    batch_size = 4
    max_epochs = 10
    seed = 42
    new_tokens = ["<new1>", "<new2>"]

    seed_everything(seed)

    wandb.init(project="textual_inversion", name="textual_inversion_training")

    dataset = TextualInversionDataset(data_root=data_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = TextualInversion(
        config=config, learning_rate=learning_rate, ckpt=ckpt, new_tokens=new_tokens
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        # gpus=1 if torch.cuda.is_available() else 0,
        accelerator="cpu",
        logger=pl.loggers.WandbLogger(),
    )
    trainer.fit(model, dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()
