import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding_mode='replicate', padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding_mode='replicate', padding=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding_mode='replicate', padding=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 9, padding_mode='replicate', padding=2),
            nn.Conv2d(64, 256, 3, padding_mode='replicate', padding=2),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 3, 9, padding_mode='replicate', padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        return self.sequence(x)