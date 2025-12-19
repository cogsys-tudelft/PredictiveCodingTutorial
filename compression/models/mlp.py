import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 384),
            nn.Tanh(),
            nn.Linear(384, 64),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 384),
            nn.Tanh(),
            nn.Linear(384, 28 * 28),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, x: torch.Tensor):
        return self.decoder(x)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

