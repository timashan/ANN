from torch import nn
from constants import *


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        R = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding="same"),
            R,
            nn.MaxPool2d((2, 2), padding=1),
            nn.Conv2d(32, 8, (3, 3), padding="same"),
            R,
            nn.MaxPool2d((2, 2), padding=1),
            nn.Conv2d(8, 8, (3, 3), padding="same"),
            R,
            nn.MaxPool2d((2, 2), padding=1),
        ).to(device)

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, (3, 3), padding="same"),
            R,
            nn.Upsample((64, 64)),
            nn.Conv2d(8, 8, (3, 3), padding="same"),
            R,
            nn.Upsample((128, 128)),
            nn.Conv2d(8, 32, (3, 3), padding="same"),
            R,
            nn.Upsample((256, 256)),
            nn.Conv2d(32, 3, (3, 3), padding="same"),
        ).to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
