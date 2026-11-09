# packages/core_engine/src/core_engine/model.py
import torch
import torch.nn as nn


class LightweightMultiModalDiT(nn.Module):
    """
    Compact DiT Backbone accepting 3D spatial tokens + Condition embeddings.
    """

    def __init__(self, in_channels: int = 3, cond_dim: int = 64, embed_dim: int = 128, depth: int = 3, heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, embed_dim)
        self.cond_proj = nn.Linear(cond_dim, embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Transformer Blocks with Scaled Dot-Product Attention
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 2,
                batch_first=True, activation="gelu"
            ) for _ in range(depth)
        ])

        self.out_proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, N, 3] Point Cloud Tokens, cond: [B, Cond_Dim]
        h = self.input_proj(x)
        t_emb = self.time_proj(t.unsqueeze(-1).float()).unsqueeze(1)
        c_emb = self.cond_proj(cond).unsqueeze(1)

        # Inject condition and timestep tokens into spatial sequence
        h = h + t_emb + c_emb

        for block in self.blocks:
            h = block(h)

        return self.out_proj(h)