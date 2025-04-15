import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_attention import CrossAttentionBlock  # adjust if needed


class DecoderBlock(nn.Module):
    """
    One up-sampling stage of the UNet decoder:
      1) Upsample previous features
      2) Concatenate with corresponding encoder skip
      3) Two Conv→Norm→ReLU layers (+ Dropout)
      4) Optional CrossAttentionBlock for semantic fusion
    """
    def __init__(self,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int,
                 use_attention: bool = False,
                 num_heads: int = 4,
                 ffn_hidden_dim: int = 256,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.use_attention = use_attention
        if use_attention:
            self.attn = CrossAttentionBlock(
                in_channels=out_channels,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout_rate=dropout_rate
            )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, tokens: torch.Tensor = None) -> torch.Tensor:
        # 1) Upsample
        x = self.upsample(x)

        # 2) Concatenate with encoder skip connection
        x = torch.cat([x, skip], dim=1)

        # 3) Convolution + Dropout
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.dropout(x)

        # 4) Optional Cross-Attention
        if self.use_attention and tokens is not None:
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H * W).permute(2, 0, 1)  # (L, B, C)
            x_attn = self.attn(x_flat, tokens)
            x = x_attn.permute(1, 2, 0).view(B, C, H, W)

        return x


class Decoder(nn.Module):
    """
    Multi-stage upsampling decoder with skip connections
    and optional semantic refinement at each stage.
    Expects [feat1, feat2, feat3] as skip features and
    bottleneck input from encoder final block.
    """
    def __init__(self,
                 use_attention: bool = True,
                 num_heads: int = 4,
                 ffn_hidden_dim: int = 256,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.block1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256,
                                   use_attention=use_attention,
                                   num_heads=num_heads,
                                   ffn_hidden_dim=ffn_hidden_dim,
                                   dropout_rate=dropout_rate)

        self.block2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128,
                                   use_attention=use_attention,
                                   num_heads=num_heads,
                                   ffn_hidden_dim=ffn_hidden_dim,
                                   dropout_rate=dropout_rate)

        self.block3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64,
                                   use_attention=use_attention,
                                   num_heads=num_heads,
                                   ffn_hidden_dim=ffn_hidden_dim,
                                   dropout_rate=dropout_rate)

    def forward(self,
                bottleneck_feat: torch.Tensor,
                encoder_features: list,
                semantic_tokens: torch.Tensor = None) -> torch.Tensor:
        """
        encoder_features: [feat1, feat2, feat3]
            feat1: (B, 64,  H*4, W*4)
            feat2: (B, 128, H*2, W*2)
            feat3: (B, 256, H,   W)
        bottleneck_feat: (B, 512, H/2, W/2)
        """
        feat1, feat2, feat3 = encoder_features

        x = self.block1(bottleneck_feat, feat3, semantic_tokens)
        x = self.block2(x, feat2, semantic_tokens)
        x = self.block3(x, feat1, semantic_tokens)

        return x
