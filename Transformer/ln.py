import torch
from torch import nn
from constants import *


class LayerNorm(nn.Module):
    def __init__(self, dims=None, eps=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)).to(device)
        self.bias = nn.Parameter(torch.zeros(1)).to(device)
        if dims:
            self.alpha = nn.Parameter(torch.ones(dims)).to(device)
            self.bias = nn.Parameter(torch.zeros(dims)).to(device)

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# LayerNorm(512)(InputEmbeddings(512, 65)(x))[0].std()
