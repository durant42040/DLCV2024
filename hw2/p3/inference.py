import argparse, os, sys, glob, json
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt_path, verbose=False):
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    model.to("cuda")

    current_vocab_size = model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.size(0)
    checkpoint_vocab_size = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"].size(0)

    if current_vocab_size > checkpoint_vocab_size:
        print(f"Extending the embedding layer to match new vocabulary size {current_vocab_size}")

        old_embed_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
        new_embed_weights = torch.cat(
            [old_embed_weights.to("cuda"),
             torch.zeros(current_vocab_size - checkpoint_vocab_size, old_embed_weights.size(1), device="cuda")],
            dim=0
        )

        sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"] = new_embed_weights

    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        if m:
            print("Missing keys:", m)
        if u:
            print("Unexpected keys:", u)

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print("Model loaded successfully.")

    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, help="Path to input json file.")
    parser.add_argument("--output_dir", type=str, help="Path to save the generated images.")
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint.")
    args = parser.parse_args()

    input_json_path = args.input_json
    outdir = args.output_dir
    ckpt_path = args.model_path
    config_path = "p3/configs/stable-diffusion/v1-inference.yaml"

    skip_save = False
    ddim_steps = 50

    ddim_eta = 0.0
    n_iter = 5
    H = 512
    W = 512
    C = 4
    f = 8
    n_samples = 5

    seed = [43, 1]
    scales = [5, 10]


    precision = "autocast"
    embedding_paths = ["<new1>_embedding.pt", "<new2>_embedding.pt"]

    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{ckpt_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(outdir, exist_ok=True)

    with open(input_json_path, "r") as file:
        input_data = json.load(file)

    start_code = None

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for key, value in input_data.items():
                    seed_everything(seed[int(key)])
                    scale = scales[int(key)]
                    model.embedding_manager.load(embedding_paths[int(key)])
                    prompts = value["prompt"]
                    for prompt_idx, prompt in enumerate(prompts):
                        prompts_batch = [prompt] * n_samples
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(n_samples * [""])
                        c = model.get_learned_conditioning(prompts_batch)
                        shape = [C, H // f, W // f]

                        output_dir = os.path.join(outdir, key, str(prompt_idx))
                        os.makedirs(output_dir, exist_ok=True)

                        image_counter = 0

                        for _ in range(n_iter):
                            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                             conditioning=c,
                                                             batch_size=n_samples,
                                                             shape=shape,
                                                             verbose=False,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=uc,
                                                             eta=ddim_eta,
                                                             x_T=start_code)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                            if not skip_save:
                                for x_sample in x_samples_ddim:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    filename = f"source{key}_prompt{prompt_idx}_{image_counter}.png"
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(output_dir, filename))
                                    image_counter += 1

    print(f"Your samples are ready and waiting for you here: \n{outdir} \n"
          f" \nEnjoy.")

if __name__ == "__main__":
    main()
