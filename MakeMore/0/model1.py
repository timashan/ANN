import torch


class BiGram:
    def __init__(self, n_vocab=27) -> None:
        self.W = torch.randn((n_vocab, n_vocab), requires_grad=True)

    def forward(self, X: torch.Tensor):
        logits = X @ self.W
        counts = logits.exp()
        self.probs = counts / counts.sum(1, keepdim=True)
        return self.probs
