# packages/diagnostics/src/diagnostics/dynamics.py
import torch
import numpy as np

class DynamicsTracker:
    """Monitors gradient dynamics, loss trajectories, and learning rates."""

    def __init__(self):
        self.history = {
            "step": [],
            "loss": [],
            "grad_norm": [],
            "lr": [],
            "snr_weights": []
        }

    def log_step(self, step: int, loss: float, model: torch.nn.Module, lr: float, snr: float = 1.0):
        # Calculate total gradient norm across trainable parameters
        total_norm = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.history["step"].append(step)
        self.history["loss"].append(loss)
        self.history["grad_norm"].append(total_norm)
        self.history["lr"].append(lr)
        self.history["snr_weights"].append(snr)

    def summary_statistics(self, last_k: int = 50) -> dict:
        recent_losses = self.history["loss"][-last_k:]
        recent_grads = self.history["grad_norm"][-last_k:]
        return {
            "mean_loss": float(np.mean(recent_losses)),
            "loss_std": float(np.std(recent_losses)),
            "mean_grad_norm": float(np.mean(recent_grads)),
            "loss_variance_stability": float(np.var(recent_losses))
        }