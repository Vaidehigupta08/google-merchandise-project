"""
train.py
========
Module 4 — CFM Training

Trains the velocity network using Conditional Flow Matching loss.

Loss:
    L = E_{t, x0, x1} [ || v_theta(x_t, t, cond) - (x1 - x0) ||^2 ]

Where:
    x_t = (1-t)*x0 + t*x1    (linear interpolation)
    x0 ~ N(0, I)
    t  ~ Uniform(0, 1)
    x1 = user embedding
    cond = cluster intent vector
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from cfm_model import CFMVelocityNet, count_parameters
from cfm_dataset import get_dataloader


# ==========================
# Paths
# ==========================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAJ_PATH  = os.path.join(BASE, "module4_Flowboost", "outputs", "trajectories.json")
MODEL_PATH = os.path.join(BASE, "module4_Flowboost", "outputs", "cfm_model.pt")
LOG_PATH   = os.path.join(BASE, "module4_Flowboost", "outputs", "train_log.json")


# ==========================
# CFM Loss
# ==========================
def cfm_loss(model, x1, condition, device):
    """
    Compute CFM MSE loss for a batch.

    Steps:
        1. Sample x0 ~ N(0, I)
        2. Sample t ~ U(0, 1)
        3. Interpolate: x_t = (1-t)*x0 + t*x1
        4. Velocity target: u_t = x1 - x0
        5. Predict: v = model(x_t, t, cond)
        6. Loss: MSE(v, u_t)
    """
    B = x1.shape[0]

    # Source noise
    x0 = torch.randn_like(x1)

    # Flow time
    t = torch.rand(B, device=device)

    # Linear interpolation (OT-CFM / straight-line flow)
    t_expand = t.view(B, 1)
    x_t = (1.0 - t_expand) * x0 + t_expand * x1

    # Velocity target (straight-line = constant velocity)
    u_t = x1 - x0

    # Predicted velocity
    v_pred = model(x_t, t, condition)

    # MSE loss
    loss = nn.functional.mse_loss(v_pred, u_t)
    return loss


# ==========================
# Training Loop
# ==========================
def train(
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 3e-4,
    emb_dim: int = 128,
    hidden_dim: int = 512,
    time_dim: int = 64,
    num_layers: int = 6,
    dropout: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Data
    loader = get_dataloader(TRAJ_PATH, batch_size=batch_size, shuffle=True)
    print(f"📦 Batches per epoch: {len(loader)}")

    # Model
    model = CFMVelocityNet(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    print(f"🧠 Model parameters: {count_parameters(model):,}")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    log = []

    print("\n🚀 Training CFM...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch in loader:
            x1        = batch["x1"].to(device)
            condition = batch["condition"].to(device)

            optimizer.zero_grad()
            loss = cfm_loss(model, x1, condition, device)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        avg_loss = np.mean(epoch_losses)
        log.append({"epoch": epoch, "loss": round(avg_loss, 6)})

        if epoch % 30 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "loss":        best_loss,
                "config": {
                    "emb_dim":    emb_dim,
                    "hidden_dim": hidden_dim,
                    "time_dim":   time_dim,
                    "num_layers": num_layers,
                    "dropout":    dropout,
                }
            }, MODEL_PATH)

    print(f"\n✅ Training complete. Best loss: {best_loss:.6f}")
    print(f"💾 Model saved → {MODEL_PATH}")

    # Save log
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"📊 Log saved → {LOG_PATH}")


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "module4_Flowboost", "outputs"), exist_ok=True)
    train()
