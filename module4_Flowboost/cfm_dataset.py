"""
cfm_dataset.py
==============
Module 4 — PyTorch Dataset

Loads trajectories.json and creates CFM training pairs:
    (x0, x1, condition, t) -> velocity target

CFM training recipe per sample:
    1. x1   = normalized user embedding (target)
    2. x0   = sample from N(0, I)        (source noise)
    3. t    = sample from Uniform(0, 1)  (flow time)
    4. x_t  = (1 - t) * x0 + t * x1    (linear interpolation)
    5. u_t  = x1 - x0                   (velocity target — straight-line flow)
    6. cond = cluster intent vector      (conditioning signal)

Model predicts v_theta(x_t, t, cond) and we minimize MSE(v_theta, u_t).
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CFMDataset(Dataset):
    """
    Dataset for Conditional Flow Matching training.

    Each item returns:
        x1        : (emb_dim,)  target embedding
        condition : (emb_dim,)  cluster intent vector
        cluster_id: int
        user_id   : str
    """

    def __init__(self, trajectories_path: str):
        with open(trajectories_path) as f:
            raw = json.load(f)

        self.data = []
        for item in raw:
            x1 = np.array(item["x1"], dtype=np.float32)
            cond = np.array(item["condition"], dtype=np.float32)

            # Safety: skip zero vectors
            if np.linalg.norm(x1) < 1e-6 or np.linalg.norm(cond) < 1e-6:
                continue

            self.data.append({
                "x1":         x1,
                "condition":  cond,
                "cluster_id": int(item["cluster_id"]),
                "user_id":    item["user_id"],
            })

        print(f"✅ CFMDataset: {len(self.data)} samples loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        return {
            "x1":         torch.tensor(item["x1"]),
            "condition":  torch.tensor(item["condition"]),
            "cluster_id": item["cluster_id"],
            "user_id":    item["user_id"],
        }


def cfm_collate(batch):
    """Stack batch items into tensors."""
    return {
        "x1":         torch.stack([b["x1"] for b in batch]),
        "condition":  torch.stack([b["condition"] for b in batch]),
        "cluster_id": [b["cluster_id"] for b in batch],
        "user_id":    [b["user_id"] for b in batch],
    }


def get_dataloader(
    trajectories_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CFMDataset(trajectories_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=cfm_collate,
        drop_last=False,
    )
