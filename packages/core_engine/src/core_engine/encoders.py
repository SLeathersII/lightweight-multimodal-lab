import torch
import torch.nn as nn

class RGBEncoder(nn.Module):
    """Stage 1A: Lightweight Vision Encoder mapping X_RGB -> z_RGB"""
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# 16x16
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).squeeze(-1).squeeze(-1)
        return self.fc(h)

class PointAutoEncoder(nn.Module):
    """Stage 1B: Spatial Point Cloud Autoencoder mapping X_3D -> z_3D -> X_hat_3D"""
    def __init__(self, num_points: int = 256, latent_dim: int = 128):
        super().__init__()
        self.num_points = num_points
        # PointNet-style 1D Conv Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 128, 1),
            nn.GELU(),
            nn.Conv1d(128, latent_dim, 1)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_points * 3)
        )

    def forward(self, x: torch.Tensor):
        # x shape: [B, N, 3] -> permute to [B, 3, N]
        h = self.encoder(x.transpose(1, 2))
        z = torch.max(h, dim=2)[0]  # Global Max Pooling -> [B, Latent_Dim]
        rec = self.decode(z)
        return z, rec

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        rec = self.decoder(z)
        return rec.view(-1, self.num_points, 3)