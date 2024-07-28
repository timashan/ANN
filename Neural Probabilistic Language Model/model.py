import torch
from torch import nn


class NGram(nn.Module):
    def __init__(
        self, n_embed: int, d_embed: int, block_size: int, n_hidden: int
    ) -> None:
        super().__init__()
        self.d_embed = d_embed
        self.block_size = block_size

        self.C = nn.Embedding(n_embed, d_embed)
        self.L1 = nn.Linear(d_embed * self.block_size, n_hidden)
        self.L2 = nn.Linear(n_hidden, n_embed)

    def forward(self, X: torch.Tensor):
        emb = self.C(X)
        h = self.L1(emb.view(-1, self.d_embed * self.block_size))
        logits = self.L2(h)
        return logits
