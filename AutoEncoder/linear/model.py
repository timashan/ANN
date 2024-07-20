from torch import nn

from constants import *


class AutoEncoder(nn.Module):
    def __init__(self, n_features=28 * 28):
        super().__init__()
        R = nn.ReLU()

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            R,
            nn.Linear(128, 64),
            R,
            nn.Linear(64, 36),
            R,
            nn.Linear(36, 18),
            R,
            nn.Linear(18, 9),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Linear(9, 18),
            R,
            nn.Linear(18, 36),
            R,
            nn.Linear(36, 64),
            R,
            nn.Linear(64, 128),
            R,
            nn.Linear(128, n_features),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
