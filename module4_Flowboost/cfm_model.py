"""
cfm_model.py
============
Module 4 — CFM Model Architecture

Implements a Conditional Flow Matching (CFM) velocity network.

How CFM works:
    - Source: x0 ~ N(0, I)        (random noise)
    - Target: x1 = user embedding  (what we want to generate)
    - Interpolation: x_t = (1-t)*x0 + t*x1
    - Velocity target: u_t = x1 - x0  (straight-line flow)
    - Model learns: v_theta(x_t, t, condition) ≈ u_t
    - At inference: integrate v from t=0 to t=1 to get x1 from x0

Conditioning:
    - condition = intent vector of user's cluster (128d)
    - This guides generation toward the cluster's behavioral space
"""

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Encodes scalar timestep t into a rich sinusoidal vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> (B, dim)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """MLP residual block with LayerNorm."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CFMVelocityNet(nn.Module):
    """
    Velocity network for Conditional Flow Matching.

    Input:
        x_t     : (B, emb_dim)   — noisy/interpolated embedding at time t
        t       : (B,)           — flow time in [0, 1]
        condition: (B, emb_dim)  — cluster intent vector (conditioning signal)

    Output:
        velocity: (B, emb_dim)   — predicted velocity field v_theta(x_t, t, c)
    """

    def __init__(
        self,
        emb_dim: int = 128,
        hidden_dim: int = 512,
        time_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.emb_dim = emb_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # Condition encoder (compress intent vector)
        self.cond_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: x_t + time + condition -> hidden_dim
        self.input_proj = nn.Linear(emb_dim + time_dim + hidden_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # Output projection -> velocity
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, emb_dim),
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:

        # Time encoding
        t_emb = self.time_embed(t)                          # (B, time_dim)

        # Condition encoding
        c_emb = self.cond_encoder(condition)                # (B, hidden_dim)

        # Fuse inputs
        h = torch.cat([x_t, t_emb, c_emb], dim=-1)        # (B, emb+time+hidden)
        h = self.input_proj(h)                              # (B, hidden_dim)

        # Residual blocks
        for block in self.blocks:
            h = block(h)

        # Predict velocity
        velocity = self.output_proj(h)                      # (B, emb_dim)
        return velocity


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
