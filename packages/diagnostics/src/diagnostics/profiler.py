# packages/diagnostics/src/diagnostics/profiler.py
import time
import os
import psutil
import torch
import torch.nn as nn


class ResourceProfiler:
    """Tracks RAM, VRAM, Active Parameters, and Latency across step evaluations."""

    @staticmethod
    def get_system_ram_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)

    @classmethod
    def profile_compression(cls, model: nn.Module, dummy_inputs: tuple, nfe_steps: int = 4) -> dict:
        device = next(model.parameters()).device
        model.eval()

        # 1. Parameter Breakdown
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        active_size_mb = (total_params * 4) / (1024 ** 2)  # FP32 Size in MB

        # 2. System RAM Pre-Check
        ram_before = cls.get_system_ram_mb()

        # 3. CUDA VRAM Reset
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        # 4. Latency & Step Evaluation
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(nfe_steps):
                _ = model(*dummy_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
        end_time = time.perf_counter()

        total_latency_ms = (end_time - start_time) * 1000
        step_latency_ms = total_latency_ms / nfe_steps
        ram_after = cls.get_system_ram_mb()
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

        return {
            "total_parameters_M": total_params / 1e6,
            "trainable_parameters_M": trainable_params / 1e6,
            "model_fp32_size_mb": active_size_mb,
            "peak_vram_mb": peak_vram_mb,
            "ram_usage_mb": ram_after - ram_before,
            "nfe_steps": nfe_steps,
            "per_step_latency_ms": step_latency_ms,
            "total_latency_ms": total_latency_ms
        }