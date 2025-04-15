import torch
import torch.nn as nn

class SemanticTokenExtractor(nn.Module):
    def __init__(self, dino_model, projected_dim=512):
        super().__init__()
        self.dino = dino_model  # frozen or trainable
        self.project = nn.Linear(self.dino.token_dim, projected_dim)

    def forward(self, x):
        """
        x: (B, 1, H, W) grayscale image (or replicate to 3 channels if needed)
        Returns: (T, B, projected_dim) tokens
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # replicate grayscale → 3-channel

        with torch.no_grad():
            tokens = self.dino.get_intermediate_layers(x, n=1)[0]  # (B, T, C)

        tokens = self.project(tokens)  # (B, T, projected_dim)
        tokens = tokens.permute(1, 0, 2)  # (T, B, projected_dim)
        return tokens
