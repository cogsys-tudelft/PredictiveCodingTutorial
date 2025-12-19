import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers: int, hidden_size: int, output_size: int):
        super(MLP, self).__init__()
        
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.gelu(layer(x))
        x = self.output_layer(x)
        return x
