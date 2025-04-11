import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, num_heads, ffn_hidden_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(in_channels, num_heads, ffn_hidden_dim)
            for _ in range(num_blocks)
        ])
        
    def forward(self, x, tokens):
        # x: shape (B, C, H, W) -> reshape to (H*W, B, C)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)
        for block in self.blocks:
            x = block(x, tokens)

        # Reshape back to (B, C, H, W)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x