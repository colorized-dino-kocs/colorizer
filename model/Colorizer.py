import torch.nn as nn

class Colorizer(nn.Module):
    pass


class OutputLayer(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(OutputLayer,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
