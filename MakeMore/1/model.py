import torch
from torch.nn import functional as F


class NGram:
    def __init__(self, d_embed=2, n_hidden=100) -> None:
        self.C = torch.randn((27, d_embed))

        self.W1 = torch.randn((6, n_hidden))
        self.b1 = torch.randn(n_hidden)

        self.W2 = torch.randn((n_hidden, 27))
        self.b2 = torch.randn(27)

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

        print(f"no.of paramters: {sum([p.nelement() for p in self.parameters])}")

    def forward(self, X: torch.Tensor):
        h = F.tanh(self.C[X].view(-1, 6) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return logits
