import os
import argparse
import torch
import torchvision.utils
from tqdm import tqdm
from UNet import UNet
from utils import beta_scheduler
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDIM(pl.LightningModule):
    def __init__(self, model, T=1000, num_steps=50, eta=0):
        super().__init__()
        self.model = model
        self.T = T
        self.register_buffer('beta', torch.tensor(beta_scheduler(T), dtype=torch.float32))
        self.alpha = 1.0 - self.beta
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.eta = eta
        self.num_steps = num_steps

    def sample(self, noise):
        x = noise
        for t in tqdm(reversed(range(1, self.T, self.T // self.num_steps)), desc="Sampling Progress",
                      total=self.num_steps):
            prev_t = (t - self.T // self.num_steps) if t != 1 else 0
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]
            alpha_bar_prev_t = self.alpha_bar[prev_t]
            eps = self.model(x, torch.full((x.shape[0],), t, device=self.device, dtype=torch.long))
            sigma_t = self.eta * torch.sqrt((1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t)
            x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0 = torch.clamp(x0, min=-1.0, max=1.0)
            z = torch.randn_like(x)
            x = torch.sqrt(alpha_bar_prev_t) * x0 + torch.sqrt(1 - alpha_bar_prev_t - sigma_t ** 2) * eps + sigma_t * z
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDIM Image Generation")
    parser.add_argument("--input_noise", type=str, required=True, help="Path to input noise tensors.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the generated images.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model weights file.")

    args = parser.parse_args()

    model_path = args.model
    output_dir = args.output_dir
    input_noise = args.input_noise

    T = 1000
    num_steps = 50
    eta = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    ddim = DDIM(model, T, num_steps, eta).to(device)

    os.makedirs(output_dir, exist_ok=True)

    noise = torch.stack([torch.load(os.path.join(input_noise, f"{i:02}.pt")) for i in range(10)]).squeeze(1)
    noise = noise.to(device)

    with torch.no_grad():
        images = ddim.sample(noise)

    for i in range(10):
        images[i] = (images[i] - images[i].min()) / (images[i].max() - images[i].min())
        torchvision.utils.save_image(images[i], os.path.join(output_dir, f"{i:02}.png"))