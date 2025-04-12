import torch.nn as nn

class MultiInputSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y):
        for layer in self.layers:
            x, y = layer(x, y)
        return x, y
