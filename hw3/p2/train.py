import os

import loralib as lora
import pytorch_lightning as pl
import timm
import torch
from dataset import ImageDataset
from decoder import Config, Decoder
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader


class ImageCaptioningTransformer(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        tokenizer,
        max_length,
        learning_rate=5e-3,
        warmup_ratio=0.3,
    ):
        super(ImageCaptioningTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_patches = 256
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        lora_save_path = f"lora_epoch_{epoch + 1}.pt"
        image_proj_save_path = f"image_proj_epoch_{epoch + 1}.pt"
        torch.save(lora.lora_state_dict(self), lora_save_path)
        torch.save(self.decoder.image_proj.state_dict(), image_proj_save_path)
        print(
            f"Saved LORA to {lora_save_path} and image_proj to {image_proj_save_path}"
        )

    def forward(self, images, captions):
        image_features = self.encoder.forward_features(images)
        return self.decoder(captions, image_features)

    def training_step(self, batch, batch_idx):
        images, captions = batch

        caption_tokens = [
            torch.tensor(self.tokenizer.encode(cap, allowed_special=["<|endoftext|>"]))
            for cap in captions
        ]

        max_len = max(len(t) for t in caption_tokens)

        input_tokens = torch.full(
            (len(caption_tokens), max_len - 1), 50256, dtype=torch.long
        )
        gt_tokens = torch.full(
            (len(caption_tokens), max_len + self.num_patches), -100, dtype=torch.long
        )

        for i, tokens in enumerate(caption_tokens):
            input_tokens[i, : len(tokens) - 1] = tokens[:-1]
            gt_tokens[i, self.num_patches : self.num_patches + len(tokens)] = tokens

        input_tokens = input_tokens.to(self.device)
        gt_tokens = gt_tokens.to(self.device)

        logits = self(images, input_tokens)
        loss = self.loss(logits.view(-1, logits.size(-1)), gt_tokens.reshape(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch

        caption_tokens = [
            torch.tensor(self.tokenizer.encode(cap, allowed_special=["<|endoftext|>"]))
            for cap in captions
        ]

        max_len = max(len(t) for t in caption_tokens)
        input_tokens = torch.full(
            (len(caption_tokens), max_len - 1), 50256, dtype=torch.long
        )
        gt_tokens = torch.full(
            (len(caption_tokens), max_len + self.num_patches), -100, dtype=torch.long
        )

        for i, tokens in enumerate(caption_tokens):
            input_tokens[i, : len(tokens) - 1] = tokens[:-1]
            gt_tokens[i, self.num_patches + 1 : self.num_patches + len(tokens)] = (
                tokens[1:]
            )

        input_tokens = input_tokens.to(self.device)
        gt_tokens = gt_tokens.to(self.device)

        logits = self(images, input_tokens)

        loss = self.loss(logits.view(-1, logits.size(-1)), gt_tokens.reshape(-1))

        if batch_idx == 0:
            with torch.no_grad():
                generated_captions = self.generate_captions(images[:4])
                for gen, ref in zip(generated_captions, captions[:4]):
                    print()
                    print(f"Generated: {gen}\nGround Truth: {ref}")

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def generate_captions(self, images: torch.Tensor) -> list:
        image_features = self.encoder.forward_features(images)
        batch_size = images.size(0)

        start_token = self.tokenizer.encode(
            "<|endoftext|>", allowed_special=["<|endoftext|>"]
        )[0]
        tokens = (
            torch.tensor([start_token])
            .unsqueeze(0)
            .to(self.device)
            .repeat(batch_size, 1)
        )

        finished_sequences = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )

        vocab_size = self.decoder.cfg.vocab_size
        token_frequencies = torch.zeros(batch_size, vocab_size, device=self.device)
        sequence_lengths = torch.zeros(
            batch_size, dtype=torch.float, device=self.device
        )

        for _ in range(self.max_length):
            if finished_sequences.all():
                break

            logits = self.decoder(tokens, image_features)

            for i in range(batch_size):
                penalty = torch.pow(1.1, token_frequencies[i])
                positive_mask = logits[i, -1, :] > 0
                negative_mask = ~positive_mask

                logits[i, -1, :][positive_mask] /= penalty[positive_mask]
                logits[i, -1, :][negative_mask] *= penalty[negative_mask]

            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)

            for i in range(batch_size):
                token_frequencies[i, next_token[i].item()] += 1
                if not finished_sequences[i]:
                    sequence_lengths[i] += 1

            tokens = torch.cat((tokens, next_token), dim=1)

            finished_sequences = finished_sequences | (
                next_token.squeeze(-1) == start_token
            )

        captions = []
        for i in range(batch_size):
            end_pos = (tokens[i] == start_token).nonzero(as_tuple=True)[0]
            if len(end_pos) > 1:
                output_tokens = tokens[i][1 : end_pos[1]]
            else:
                output_tokens = tokens[i][1:]
            caption = self.tokenizer.decode(output_tokens.tolist())
            captions.append(caption)

        return captions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0
        )
        return optimizer


def main():
    os.environ["WANDB_API_KEY"] = "cea091384aba595fb3f51ee372ed755a0ac3ba5d"
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = timm.create_model(
        "vit_large_patch14_clip_224.openai",
        pretrained=True,
        num_classes=0,
    ).to(device)
    decoder = Decoder(Config(checkpoint="hw3_data/p2_data/decoder_model.bin")).to(
        device
    )
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    data_config = timm.data.resolve_model_data_config(encoder)
    transforms = timm.data.create_transform(**data_config, is_training=True)

    encoder.eval()

    train_dataset = ImageDataset(
        "hw3_data/p2_data/images/train", "hw3_data/p2_data/train.json", transforms
    )
    val_dataset = ImageDataset(
        "hw3_data/p2_data/images/val", "hw3_data/p2_data/val.json", transforms
    )

    logger = WandbLogger(project="image-captioning", name="transformer-training")

    model = ImageCaptioningTransformer(
        encoder,
        decoder,
        tokenizer,
        max_length=50,
        learning_rate=6e-3,
        warmup_ratio=0.3,
    )
    # model.decoder.image_proj.load_state_dict(
    #     torch.load("checkpoints/image_proj.pt", map_location=device),
    #     strict=False,
    # )
    # model.load_state_dict(
    #     torch.load("checkpoints/lora.pt", map_location=device), strict=False
    # )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    for param in model.parameters():
        param.requires_grad = True

    lora.mark_only_lora_as_trainable(model)

    for param in model.decoder.image_proj.parameters():
        param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    print(
        "Total Params:", sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    trainer = pl.Trainer(
        max_epochs=8,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
