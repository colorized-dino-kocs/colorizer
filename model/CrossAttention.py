class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, ffn_hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_channels)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.ln2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden_dim, in_channels)
        )
    
    def forward(self, x, tokens):
        x_norm = self.ln1(x)
        attn_out, _ = self.cross_attn(query=x_norm, key=tokens, value=tokens)
        x += attn_out
        
        x_norm2 = self.ln2(x)
        ffn_out = self.ffn(x_norm2)
        x += ffn_out
        
        return x