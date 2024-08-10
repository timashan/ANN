import torch

import sys

sys.path.insert(0, "..")
import nn


class NGram:
    def __init__(self, block_size=3, vocab_size=27, n_embd=10, n_hidden=100) -> None:

        self.seq = nn.Sequential(
            [
                nn.Embedding(vocab_size, n_embd),
                #
                nn.Flatten(2),
                nn.Linear(n_embd * 2, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                #
                nn.Flatten(2),
                nn.Linear(n_hidden * 2, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                #
                nn.Flatten(2),
                nn.Linear(n_hidden * 2, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.Tanh(),
                #
                nn.Linear(n_hidden, vocab_size, bias=False),
                # nn.BatchNorm1d(vocab_size),
            ]
        )
        self.layers = self.seq.layers

        with torch.no_grad():
            self.layers[-1].weight *= 0.1

        self.parameters = [p for layer in self.layers for p in layer.parameters()]
        print(sum(p.nelement() for p in self.parameters))
        for p in self.parameters:
            p.requires_grad = True

    def __call__(self, x):
        x = self.seq(x)
        return x

    def set_training(self, isTrain=True):
        for layer in self.layers:
            layer.training = isTrain
