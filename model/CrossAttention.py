import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)

        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, dropout=dropout_rate)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_dim, in_channels),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, tokens):
        # x: (L, B, C), tokens: (T, B, C)
        x_norm = self.ln1(x)
        attn_out, _ = self.cross_attn(query=x_norm, key=tokens, value=tokens)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out

        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out
        return x, tokens
