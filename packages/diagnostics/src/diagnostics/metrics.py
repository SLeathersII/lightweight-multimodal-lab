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

class ModalityDiscrepancyEvaluator:
    """Probes failure modes in one modality that are absent in another."""

    @staticmethod
    def corrupt_3d_point_cloud(pc: torch.Tensor, drop_ratio: float = 0.75) -> torch.Tensor:
        """Simulates sensor sparsity / LiDAR beam drop out."""
        B, N, C = pc.shape
        keep_count = int(N * (1.0 - drop_ratio))
        indices = torch.randperm(N)[:keep_count]
        sparse_pc = pc[:, indices, :]
        # Pad back to shape for standard metric calculation
        padding = torch.zeros(B, N - keep_count, C, device=pc.device)
        return torch.cat([sparse_pc, padding], dim=1)

    @staticmethod
    def corrupt_rgb_occlusion(img: torch.Tensor, mask_ratio: float = 0.50) -> torch.Tensor:
        """Simulates camera lens blinding / visual occlusion."""
        masked_img = img.clone()
        _, _, H, W = img.shape
        mask_w = int(W * mask_ratio)
        masked_img[:, :, :, :mask_w] = 0.0  # Zero out left half of image
        return masked_img