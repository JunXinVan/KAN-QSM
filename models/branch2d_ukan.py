# -*- coding: utf-8 -*-
"""
2D branch (stable v0.9) for each plane:
- Simple2DBranch: light encoder -> CSAR-Lite (fuse L slices) -> 1x1 conv
  Input : [B, L, 1, P, P]
  Output: [B, 1, P, P]   (central-slice prediction for this plane)

A separate UKAN wrapper will be provided in the next round to replace this branch
once your UKAN code is in place. This file is fully runnable without UKAN.

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .csar_lite import CSARLite


class Simple2DBranch(nn.Module):
    """
    Minimal and stable 2D branch:
      - Per-slice 2D encoder (shared for all L slices)
      - CSAR-Lite to fuse across slice axis L
      - 1x1 conv head to regression
    """
    def __init__(
        self,
        L: int = 11,
        in_ch: int = 1,
        base: int = 64,
        csar_rank: int = 4,
        csar_T: float = 1.0,
        sem_hidden_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.L = int(L)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.csar = CSARLite(L=L, C=base, slice_rank=csar_rank, T=csar_T, sem_hidden_ratio=sem_hidden_ratio)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs: [B, L, 1, P, P]
        return: [B, 1, P, P]
        """
        B, L, C, H, W = xs.shape
        assert L == self.L and C == 1, f"Expected [B,{self.L},1,P,P], got {xs.shape}"
        x = xs.view(B * L, 1, H, W)             # merge slices
        feat = self.enc(x)                      # [B*L, base, P, P]
        feat = feat.view(B, L, -1, H, W)        # [B, L, base, P, P]
        fused = self.csar(feat)                 # [B, base, P, P]
        y = self.out(fused)                     # [B, 1, P, P]
        return y
