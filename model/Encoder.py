import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_pool: bool = True, dropout_rate=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

        self.with_pool = with_pool

        if self.with_pool:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.net(x)

        if not self.with_pool:
            return x

        x_down = self.pool(x)
        return x, x_down


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = EncoderBlock(1, 64)
        self.down2 = EncoderBlock(64, 128)
        self.down3 = EncoderBlock(128, 256)
        self.down4 = EncoderBlock(256, 512, with_pool=False)

    def forward(self, x):
        feat1, x = self.down1(x)
        feat2, x = self.down2(x)
        feat3, x = self.down3(x)
        x = self.down4(x)

        return feat1, feat2, feat3, x
