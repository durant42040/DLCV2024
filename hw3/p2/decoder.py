import collections
import math

import loralib as lora
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.image_feature_size = 1024
        self.checkpoint = checkpoint


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=16)
        # self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=16)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer(
            "bias", torch.tril(torch.ones(size, size)).view(1, 1, size, size)
        )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        attn_mask = torch.tril(torch.ones(T, T)).to(x.device)
        attn_mask[:257, 257:] = 0
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, T, T)
        attn_mask = attn_mask == 0

        att = att.masked_fill(attn_mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.mlp = nn.Sequential(
            collections.OrderedDict(
                [
                    # ("c_fc", nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
                    ("c_fc", lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=16)),
                    ("act", nn.GELU(approximate="tanh")),
                    # ("c_proj", nn.Linear(4 * cfg.n_embd, cfg.n_embd)),
                    ("c_proj", lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=16)),
                ]
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                h=nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=nn.LayerNorm(cfg.n_embd),
            )
        )
        # self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, r=16, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # self.image_proj = nn.Linear(cfg.image_feature_size, cfg.n_embd)
        self.image_proj = nn.Sequential(
            nn.Linear(cfg.image_feature_size, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.n_embd),
        )
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [".c_attn.weight", ".c_fc.weight", ".c_proj.weight"]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x: Tensor, image_features: Tensor):
        image_features = self.image_proj(image_features)

        x = torch.narrow(x, 1, 0, min(x.size(1), self.block_size))
        pos = torch.arange(x.size()[1], dtype=torch.long, device=x.device).unsqueeze(0)

        image_pos = torch.arange(
            image_features.size()[1], dtype=torch.long, device=x.device
        ).unsqueeze(0)

        x = self.transformer.wte(x) + self.transformer.wpe(pos)
        image_features = image_features + self.transformer.wpe(image_pos)

        x = torch.cat((image_features, x), dim=1)

        for block in self.transformer.h:
            x = block(x)

        x = self.lm_head(self.transformer.ln_f(x))
        return x


if __name__ == "__main__":
    model = Decoder(Config())
