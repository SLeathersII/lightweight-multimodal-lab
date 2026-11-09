# packages/diagnostics/src/diagnostics/metrics.py
import torch

class FidelityEvaluator:
    """Computes cross-modal generation fidelity metrics."""

    @staticmethod
    def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
        """
        Computes symmetric Chamfer Distance between two 3D point cloud sets.
        pc1: [B, N, 3], pc2: [B, M, 3]
        """
        # Pairwise distance matrix: [B, N, M]
        dist_matrix = torch.cdist(pc1, pc2, p=2) ** 2

        # Min distances per point
        min_dist_pc1 = torch.min(dist_matrix, dim=2)[0]  # [B, N]
        min_dist_pc2 = torch.min(dist_matrix, dim=1)[0]  # [B, M]

        # Mean over point clouds
        chamfer = torch.mean(min_dist_pc1, dim=1) + torch.mean(min_dist_pc2, dim=1)
        return torch.mean(chamfer)

    @staticmethod
    def spatial_alignment_mse(pred_layout: torch.Tensor, target_layout: torch.Tensor) -> torch.Tensor:
        """Evaluates spatial bounding box / coordinate alignment accuracy."""
        return torch.mean((pred_layout - target_layout) ** 2)