import torch
from torch import nn
from torch.nn import functional as F

from constants import *


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout=0):
        super().__init__()
        self.d_model = d_model

        self.h = h
        self.d_k = d_model // h

        self.w_k = nn.Linear(d_model, d_model, device=device)
        self.w_q = nn.Linear(d_model, d_model, device=device)
        self.w_v = nn.Linear(d_model, d_model, device=device)
        self.w_o = nn.Linear(d_model, d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, device=device))
        )

    def forward(self, x):
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        B, T, C = x.shape
        split_head = lambda x: x.view(B, T, self.h, self.d_k).transpose(1, 2)
        q = split_head(q)
        k = split_head(k)
        v = split_head(v)

        # attention
        B, H, T, C = k.shape
        att_scores = q @ k.transpose(-1, -2) * C**-0.5

        tril = torch.tril(torch.ones(T, T).to(device))
        att_scores = att_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        att_scores = att_scores.softmax(dim=-1)

        if self.dropout:
            att_scores = self.dropout(att_scores)

        att_scores = att_scores @ v

        out = att_scores.transpose(1, 2).contiguous().view(B, T, H * C)
        out = self.w_o(out)
        return out
