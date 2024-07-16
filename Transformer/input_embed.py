import torch
from torch import nn
from constants import *


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.E = nn.Embedding(vocab_size, d_model, device=device)

    def forward(self, x):
        return self.E(x)


# Ex
x = torch.randint(0, 65, size=(5, 8), device=device)
# print(torch.randn(65, 2)[x].shape)
x = InputEmbedding(65, 512)(x)
# print(x.shape)
