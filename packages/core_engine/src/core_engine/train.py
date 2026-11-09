# packages/core_engine/src/core_engine/train.py
import torch
import torch.optim as optim
from core_engine.model import LightweightMultiModalDiT
from core_engine.dataset import get_clevr_dataloader
from diagnostics.profiler import ResourceProfiler
from diagnostics.metrics import FidelityEvaluator
from diagnostics.dynamics import DynamicsTracker


def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CLEVR-3D investigative suite on device: {device}")

    # 1. Setup Data, Architecture & Dynamics Tracking
    B, N_POINTS, COND_DIM = 16, 256, 64
    dataloader = get_clevr_dataloader(batch_size=B, num_points=N_POINTS, cond_dim=COND_DIM)

    model = LightweightMultiModalDiT(
        in_channels=3,
        cond_dim=COND_DIM,
        embed_dim=128,
        depth=3,
        heads=4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    tracker = DynamicsTracker()

    # 2. Benchmark Model Overhead
    sample_clean_pc, sample_cond = next(iter(dataloader))
    dummy_x = sample_clean_pc.to(device)
    dummy_t = torch.randint(0, 1000, (B,)).to(device)
    dummy_c = sample_cond.to(device)

    profile_report = ResourceProfiler.profile_compression(model, (dummy_x, dummy_t, dummy_c), nfe_steps=4)
    print("\n--- Model Compression & Resource Profile ---")
    for k, v in profile_report.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 3. Fast Training Loop over CLEVR-3D Geometry & Spatial Conditions
    print("\n--- Training Loop & Density Estimation Dynamics ---")
    model.train()

    for step, (clean_pc, cond) in enumerate(dataloader, start=1):
        clean_pc = clean_pc.to(device)
        cond = cond.to(device)

        optimizer.zero_grad()

        # Sample random timesteps t
        t = torch.randint(0, 1000, (B,)).to(device)
        noise = torch.randn_like(clean_pc)

        # Standard Linear Noise Schedule (Alpha-bar)
        alpha_t = (1.0 - (t.float() / 1000.0)).view(B, 1, 1)
        noisy_pc = torch.sqrt(alpha_t) * clean_pc + torch.sqrt(1.0 - alpha_t) * noise

        # Score-Matching / Noise Prediction Pass
        pred_noise = model(noisy_pc, t, cond)
        loss = torch.mean((pred_noise - noise) ** 2)

        loss.backward()
        current_lr = scheduler.get_last_lr()[0]
        tracker.log_step(step, loss.item(), model, current_lr)

        optimizer.step()
        scheduler.step()

        if step % 15 == 0:
            stats = tracker.summary_statistics(last_k=15)
            print(
                f"Step {step:03d}/{len(dataloader)} | Loss: {stats['mean_loss']:.5f} | Grad Norm: {stats['mean_grad_norm']:.4f} | LR: {current_lr:.6f}")

    # 4. Evaluate Generation Fidelity (Chamfer Distance on Real CLEVR Data)
    model.eval()
    with torch.no_grad():
        # Test 1-step conditional noise prediction sample
        generated_noise = model(noisy_pc, t, cond)
        # Reconstruct approximate clean point cloud
        pred_clean_pc = (noisy_pc - torch.sqrt(1.0 - alpha_t) * generated_noise) / torch.sqrt(alpha_t)

        cd_score = FidelityEvaluator.chamfer_distance(clean_pc, pred_clean_pc)
        print(f"\nGenerative Quality -> CLEVR-3D Chamfer Distance: {cd_score.item():.6f}")


if __name__ == "__main__":
    run_experiment()