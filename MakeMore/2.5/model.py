import torch
import nn


class NGram:
    def __init__(self, block_size=3, vocab_size=27, n_embd=10, n_hidden=100) -> None:

        self.layers = [
            nn.Embedding(vocab_size, n_embd),
            nn.Linear(n_embd * block_size, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, vocab_size, bias=False),
            nn.BatchNorm1d(vocab_size),
        ]

        with torch.no_grad():
            self.layers[-1].gamma *= 0.1
            # for layer in self.layers:
            #     if isinstance(layer, nn.Linear):
            #         layer.weight *= 5 / 3

        self.parameters = [p for layer in self.layers for p in layer.parameters()]
        print(sum(p.nelement() for p in self.parameters))
        for p in self.parameters:
            p.requires_grad = True

    def __call__(self, X):
        emb = self.layers[0](X)  # embed
        x = emb.view(emb.shape[0], -1)  # concat
        for layer in self.layers[1:]:
            x = layer(x)
        return x
