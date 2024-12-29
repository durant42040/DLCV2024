import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from decoder import Config, Decoder
from PIL import Image
from scipy.ndimage import zoom
from tokenizer import BPETokenizer
from torch import nn
from torchvision import transforms


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

    def generate_captions(self, images: torch.Tensor):
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

        attention_weights_per_step = []
        for _ in range(self.max_length):
            if finished_sequences.all():
                break

            logits = self.decoder(tokens, image_features)
            attention_weights_per_step.append(
                [w.detach().cpu().numpy() for w in self.decoder.attention_weights]
            )

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

        return captions, attention_weights_per_step


def create_high_res_attention_map(attention_weights, upscale_factor=14):
    """
    Create a high resolution attention map from patch-based attention weights.

    Args:
        attention_weights (np.ndarray): Attention weights of shape (16, 16)
        upscale_factor (int): Factor to upscale the attention map (16*14 = 224)

    Returns:
        np.ndarray: Upscaled attention map of shape (224, 224)
    """
    # Smooth upscaling using cubic interpolation
    attention_map = zoom(attention_weights, upscale_factor, order=3)

    # Normalize to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    attention_map = np.clip(attention_map, None, 0.45)
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )

    return attention_map


def visualize_attention_overlay(image_path, model, tokenizer, device, save_dir=None):
    """
    Create high-resolution attention visualization overlaid on the original image.

    Args:
        image_path (str): Path to input image
        model (nn.Module): Image captioning model
        tokenizer: Tokenizer for text processing
        device (str): Device to run model on
        save_dir (str, optional): Directory to save visualizations
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((224, 224), Image.Resampling.LANCZOS)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # Generate caption and get attention weights
    model.eval()
    with torch.no_grad():
        captions, attention_weights_list = model.generate_captions(image_tensor)

    caption = captions[0].split("<|endoftext|>")[0].split("<|")[0].strip()
    words = caption.split()

    # Process attention weights
    attention_weights = []
    for step_weights in attention_weights_list:
        layer_weights = []
        for layer_weight in step_weights:
            layer_weights.append(layer_weight[0])
        attention_weights.append(layer_weights)

    # Create visualizations for each layer
    n_layers = len(attention_weights[0])

    for layer_idx in range(n_layers):
        # Create figure for this layer
        n_words = len(words)
        n_cols = min(4, n_words)
        n_rows = (n_words + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle(
            f"Layer {layer_idx + 1} Attention Visualization\nCaption: {caption}",
            fontsize=16,
            y=1.02,
        )

        # Plot attention overlay for each word
        for word_idx, word in enumerate(words):
            # Get attention weights for current word
            token_idx = word_idx + 256 + 1  # 256 image patches + 1 start token
            attention = attention_weights[word_idx][
                layer_idx
            ]  # [n_heads, seq_len, seq_len]

            # Average attention across heads
            avg_attention = attention[:, token_idx, :256].mean(
                0
            )  # Average across heads
            attention_map = avg_attention.reshape(16, 16)

            # Create high-resolution attention map
            hi_res_attention = create_high_res_attention_map(attention_map)

            # Create subplot
            ax = plt.subplot(n_rows, n_cols, word_idx + 1)

            # Display original image
            ax.imshow(original_image)

            # Overlay attention heatmap with custom colormap
            attention_mask = ax.imshow(
                hi_res_attention, cmap="jet", alpha=0.8, interpolation="lanczos"
            )

            # Add word and attention stats
            ax.set_title(f'"{word}"\nMax Attention: {avg_attention.max():.3f}')
            ax.axis("off")

            # Add colorbar
            plt.colorbar(attention_mask, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save or show figure
        if save_dir:
            plt.savefig(
                f"{save_dir}/attention_layer_{layer_idx + 1}.png",
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.show()
        plt.close()


def visualize_head_attention_hires(
    image_path, model, tokenizer, device, layer_idx=0, word_idx=0, save_dir=None
):
    """
    Visualize high-resolution attention maps for each attention head.
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((224, 224), Image.Resampling.LANCZOS)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    # Generate caption and get attention weights
    model.eval()
    with torch.no_grad():
        captions, attention_weights_list = model.generate_captions(image_tensor)

    caption = captions[0].split("<|endoftext|>")[0].split("<|")[0].strip()
    words = caption.split()

    # Get attention weights for specified word and layer
    attention = attention_weights_list[word_idx][layer_idx][0]
    n_heads = attention.shape[0]

    # Create visualization
    n_cols = 4
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(20, 5 * n_rows))
    fig.suptitle(
        f'Layer {layer_idx + 1}, Word "{words[word_idx]}" - Attention Heads\n'
        f"Caption: {caption}",
        fontsize=16,
        y=1.02,
    )

    # Plot attention for each head
    for head_idx in range(n_heads):
        ax = plt.subplot(n_rows, n_cols, head_idx + 1)

        # Get attention weights for current head
        token_idx = word_idx + 256 + 1
        head_attention = attention[head_idx, token_idx, :256]
        attention_map = head_attention.reshape(16, 16)

        # Create high-resolution attention map
        hi_res_attention = create_high_res_attention_map(attention_map)

        # Display original image
        ax.imshow(original_image)

        # Overlay attention heatmap
        attention_mask = ax.imshow(
            hi_res_attention, cmap="magma", alpha=0.7, interpolation="lanczos"
        )

        ax.set_title(f"Head {head_idx + 1}\nMax Attention: {head_attention.max():.3f}")
        ax.axis("off")

        # Add colorbar
        plt.colorbar(attention_mask, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Save or show figure
    if save_dir:
        plt.savefig(
            f"{save_dir}/head_attention_layer_{layer_idx + 1}_word_{word_idx}.png",
            bbox_inches="tight",
            dpi=300,
        )
    else:
        plt.show()
    plt.close()


# Example usage:
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = timm.create_model(
        "vit_large_patch14_clip_224.openai",
        pretrained=True,
        num_classes=0,
    ).to(device)
    decoder = Decoder(Config(checkpoint="hw3_data/p2_data/decoder_model.bin")).to(
        device
    )
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")

    # decoder.eval()
    encoder.eval()

    model = ImageCaptioningTransformer(
        encoder=encoder, decoder=decoder, tokenizer=tokenizer, max_length=50
    ).to(device)

    model.decoder.image_proj.load_state_dict(
        torch.load("checkpoints/image_proj.pt", map_location=device),
        strict=False,
    )
    model.load_state_dict(
        torch.load("checkpoints/lora.pt", map_location=device), strict=False
    )
    model.eval()

    # Visualize attention for all words
    torch.manual_seed(42)

    # Initialize your model, tokenizer, etc.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    import os

    os.makedirs("attention_vis", exist_ok=True)

    # Visualize attention for all words
    visualize_attention_overlay(
        image_path="hw3_data/p2_data/images/val/000000001086.jpg",
        model=model,
        tokenizer=tokenizer,
        device=device,
        save_dir="attention_vis",
    )

    # Visualize individual head attention for a specific word
    visualize_head_attention_hires(
        image_path="hw3_data/p3_data/images/umbrella.jpg",
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=0,
        word_idx=0,
        save_dir="attention_vis",
    )


if __name__ == "__main__":
    main()
