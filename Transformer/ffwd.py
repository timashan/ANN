import torch
from torch import nn
from constants import *


class FFWD(nn.Module):
    def __init__(self, d_model: int, n_embed=None, dropout=0):
        super().__init__()
        self.d_model = d_model
        n_embed = n_embed if n_embed else d_model * 4

        self.net = nn.Sequential(
            nn.Linear(d_model, n_embed, device=device),
            nn.ReLU(),
            nn.Linear(n_embed, d_model, device=device),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        assert (
            x.shape[-1] == self.d_model
        ), f"Invalid dimenstionality. Expected: {d_model} got {x.shape[-1]}."

        return self.net(x)


# Ex
x = torch.randn(5, 8, 512, device=device)
x = FFWD(512)(x)
# print(x.shape)
