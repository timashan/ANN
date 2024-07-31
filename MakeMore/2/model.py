import torch
from torch.nn import functional as F


class NGram:
    def __init__(self, vocab_size=27, n_embed=2, block_size=3, n_hidden=100) -> None:
        self.C = torch.randn((vocab_size, n_embed))

        W1_fan_in = n_embed * block_size
        self.W1 = torch.randn((W1_fan_in, n_hidden)) * (5 / 3) / (W1_fan_in**0.5)  # 0.2
        self.b1 = torch.randn(n_hidden) * 0.01

        self.W2 = torch.randn((n_hidden, vocab_size)) * 0.01
        self.b2 = torch.randn(vocab_size) * 0

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

        print(f"no.of paramters: {sum([p.nelement() for p in self.parameters])}")

    def forward(self, X: torch.Tensor):
        h = F.tanh(self.C[X].view(-1, 6) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits
