# -*- coding: utf-8 -*-
"""
CSAR-Lite (stable v0.9): Cross-Slice Attention for Regression (2.5D)
- Semantic attention (channel-wise SE-style, MLP)
- Slice attention (low-rank diagonal variance Gaussian weights along slice axis, expectation only)
- Both modules are lightweight and numerically stable; no sampling; suitable for regression.

Input/Output:
  xs  : [B, L, C, P, P]  -> slices stacked along axis L per 2D branch
  out : [B, C, P, P]

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAttn(nn.Module):
    """
    Channel-wise (semantic) attention using a small MLP.
    Optionally you can later inject a KAN-MLP by替换self.proj即可。
    """
    def __init__(self, channels: int, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        h = max(4, int(channels * hidden_ratio))
        self.proj = nn.Sequential(
            nn.Linear(channels, h),
            nn.GELU(),
            nn.Linear(h, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, P, P]
        """
        b, c, h, w = x.shape
        gap = x.mean(dim=(2, 3))           # [B,C]
        wgt = torch.sigmoid(self.proj(gap))# [B,C]
        return x * wgt.view(b, c, 1, 1)


class SliceAttnLowRank(nn.Module):
    """
    Slice attention along L with a diagonal+low-rank variance model (no sampling).
    We only use the diagonal of P P^T (i.e., sum of squares per slice index) for stability.
    Weights:
        w_i ∝ exp( - (i - mu)^2 / (2 * var_i * T) ),  then normalized along L.
    Parameters:
      L     : number of slices
      rank  : low-rank width for P ∈ R^{L×rank};  var_i = sum_j P[i,j]^2 + softplus(D[i]) + eps
      T     : temperature for test-time calibration (TTA-lite friendly)
    """
    def __init__(self, L: int, rank: int = 4, T: float = 1.0) -> None:
        super().__init__()
        self.L = int(L)
        self.rank = int(rank)
        self.T = float(T)
        # Learnable center and diagonal variance components
        self.mu = nn.Parameter(torch.tensor(float(L // 2)))
        self.P  = nn.Parameter(torch.randn(L, rank) * 0.05)
        self.D  = nn.Parameter(torch.full((L,), -2.0))  # logit-ish; softplus -> positive
        self.eps = 1e-5

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs: [B, L, C, P, P]
        returns [B, C, P, P]
        """
        B, L, C, H, W = xs.shape
        device = xs.device
        idx = torch.arange(L, device=device, dtype=xs.dtype)
        # var_i = sum_j P[i,j]^2 + softplus(D[i]) + eps
        var = (self.P ** 2).sum(dim=1) + F.softplus(self.D) + self.eps  # [L]
        # Gaussian weights along slices
        w = torch.exp(-0.5 * ((idx - self.mu) ** 2) / (var * self.T))
        w = (w / (w.sum() + 1e-6)).view(1, L, 1, 1, 1)                  # [1,L,1,1,1]
        x = (xs * w).sum(dim=1)  # [B,C,P,P]
        return x


class CSARLite(nn.Module):
    """
    xs -> slice attention (L axis) -> semantic attention (channel)
    """
    def __init__(self, L: int, C: int, slice_rank: int = 4, T: float = 1.0, sem_hidden_ratio: float = 0.5) -> None:
        super().__init__()
        self.slice = SliceAttnLowRank(L=L, rank=slice_rank, T=T)
        self.seman = SemanticAttn(channels=C, hidden_ratio=sem_hidden_ratio)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs: [B, L, C, P, P]
        returns [B, C, P, P]
        """
        x = self.slice(xs)
        x = self.seman(x)
        return x
