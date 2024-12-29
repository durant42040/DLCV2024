import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration


def create_high_res_attention_map(attention_weights, upscale_factor=14):
    """
    Create a high-resolution attention map from patch-based attention weights.

    Args:
        attention_weights (np.ndarray): Attention weights of shape (16, 16).
        upscale_factor (int): Factor to upscale the attention map (e.g., 16*14 = 224).

    Returns:
        np.ndarray: Upscaled attention map of shape (224, 224).
    """
    # Smooth upscaling using cubic interpolation
    attention_map = zoom(attention_weights, upscale_factor, order=3)

    # Normalize to [0, 1]
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )
    return attention_map


def visualize_attention_overlay(image_path, model, processor, device, save_dir=None):
    """
    Create high-resolution attention visualization overlaid on the original image.

    Args:
        image_path (str): Path to the input image.
        model: Llava model for generating captions.
        processor: Llava processor for image processing.
        device (str): Device to run the model on.
        save_dir (str, optional): Directory to save visualizations.
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_image_resized = original_image.resize((224, 224), Image.Resampling.LANCZOS)

    # Preprocess image for the model
    inputs = processor(
        images=original_image,
        text="USER: <image>\nWrite a description for the photo in one sentence. ASSISTANT:",
        return_tensors="pt",
    ).to(device)

    # Generate caption and get attention weights
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, output_attentions=True, return_dict_in_generate=True
        )
        captions = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
        attention_weights = outputs.attentions  # List of attention layers

    caption = captions[0].strip()
    words = caption.split()

    # Process attention weights
    attention_map_list = []
    for layer_attention in attention_weights:
        # Extract attention for CLS token and image patches
        cls_attention = layer_attention[0, :, 0, 1:257]  # [batch, head, cls, patches]
        avg_attention = cls_attention.mean(1)  # Average across heads
        attention_map = avg_attention.reshape(16, 16)  # Assuming 16x16 patches
        attention_map_list.append(attention_map)

    # Create visualizations for each attention layer
    n_layers = len(attention_map_list)
    for layer_idx, attention_map in enumerate(attention_map_list):
        # Create high-resolution attention map
        hi_res_attention = create_high_res_attention_map(attention_map)

        # Plot original image with attention overlay
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(original_image_resized)
        attention_mask = ax.imshow(hi_res_attention, cmap="jet", alpha=0.5)

        # Add title and colorbar
        ax.set_title(
            f'Layer {layer_idx + 1} Attention Overlay\nCaption: "{caption}"',
            fontsize=14,
        )
        ax.axis("off")
        plt.colorbar(attention_mask, ax=ax, fraction=0.046, pad=0.04)

        # Save or show visualization
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f"layer_{layer_idx + 1}_attention.png"),
                bbox_inches="tight",
                dpi=300,
            )
        else:
            plt.show()
        plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, load_in_4bit=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    # Path to the input image
    image_path = "hw3_data/p3_data/images/umbrella.jpg"

    # Directory to save visualizations
    save_dir = "attention_vis"

    # Visualize attention overlay
    visualize_attention_overlay(
        image_path=image_path,
        model=model,
        processor=processor,
        device=device,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
