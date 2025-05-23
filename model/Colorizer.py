import torch.nn as nn

from model import Encoder, Bottleneck, Decoder, SemanticTokenExtractor
from .TokenExtractor import SemanticTokenExtractor


class Colorizer(nn.Module):
    def __init__(self, dino_model):
        super().__init__()

        self.encoder = Encoder()
        self.token_extractor = SemanticTokenExtractor(dino_model, projected_dim=512)

        self.bottleneck = Bottleneck(
            in_channels=512,
            num_heads=4,
            ffn_hidden_dim=1024,
            num_blocks=2,
            dropout_rate=0.1
        )

        self.decoder = Decoder(
            use_attention=True,        # fuse semantics at every stage
            num_heads=4,
            ffn_hidden_dim=1024,
            dropout_rate=0.1
        )
        self.out = OutputLayer()

    def forward(self, x):
        # x: grayscale image, shape (B, 1, H, W)

        *residual_features, encoded = self.encoder(x)
        semantic_tokens = self.token_extractor(x)
        bottleneck = self.bottleneck(encoded, semantic_tokens)
        decoded = self.decoder(bottleneck, residual_features, semantic_tokens)

        return self.out(decoded)




class OutputLayer(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        return x
