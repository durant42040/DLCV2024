import argparse
import json

import timm
import torch
from dataset import InferenceDataset
from decoder import Config, Decoder
from tokenizer import BPETokenizer
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class ImageCaptioningTransformer(nn.Module):
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images, captions):
        image_features = self.encoder.forward_features(images)
        return self.decoder(captions, image_features)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output"
    )
    parser.add_argument(
        "--decoder_path", type=str, required=True, help="Path to the decoder model"
    )

    args = parser.parse_args()

    img_path = args.img_path
    output_path = args.output_path
    decoder_path = args.decoder_path

    max_length = 50
    batch_size = 16
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = timm.create_model(
        "vit_large_patch14_clip_224.openai",
        pretrained=True,
        num_classes=0,
    ).to(device)
    decoder = Decoder(Config(checkpoint=decoder_path)).to(device)
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    # decoder.eval()
    encoder.eval()

    model = ImageCaptioningTransformer(
        encoder=encoder, decoder=decoder, tokenizer=tokenizer, max_length=max_length
    ).to(device)

    model.decoder.image_proj.load_state_dict(
        torch.load("checkpoints/image_proj.pt", map_location=device),
        strict=False,
    )
    model.load_state_dict(
        torch.load("checkpoints/lora.pt", map_location=device), strict=False
    )
    model.eval()

    data_config = timm.data.resolve_model_data_config(encoder)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    val_dataset = InferenceDataset(img_path, transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pred_data = {}
    with torch.no_grad():
        for batch_idx, (image_id, images) in enumerate(
            tqdm(val_loader, desc="Generating captions")
        ):
            images = images.to(device)

            generated_captions = model.generate_captions(images)

            for j, img_id in enumerate(image_id):
                pred_data[f"{img_id.item():012}"] = (
                    generated_captions[j].split("<|endoftext|>")[0].split("<|")[0]
                )

    with open(output_path, "w") as f:
        json.dump(pred_data, f, indent=4)


if __name__ == "__main__":
    main()
