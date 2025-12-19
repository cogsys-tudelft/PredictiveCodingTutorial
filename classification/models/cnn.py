import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels: int, output_size: int):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(1),
        )

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.gelu(self.layer1(x))
        x = F.gelu(self.layer2(x))
        x = F.gelu(self.layer3(x))

        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return x

