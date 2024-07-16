import torch
from torch import nn
from constants import *


class PositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.E = nn.Embedding(block_size, d_model, device=device)

    def forward(self, x: torch.Tensor):
        assert (
            x.shape[-1] == self.d_model
        ), f"Invalid dimenstionality. Expected: {d_model} got {x.shape[-1]}."

        return x + self.E(torch.arange(self.block_size, device=device))  # broadcast


# Ex
x = torch.randn(5, 8, 512, device=device)
x = PositionalEmbedding(8, 512)(x)
# print(x.shape)
