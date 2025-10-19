# -*- coding: utf-8 -*-
"""
Tri-plane 2D -> 3D fusion head (stable v0.9)

- Embed each plane's 2D prediction [B,1,P,P] into a 3D pseudo-volume [B,1,P,P,P]
  by broadcasting the 2D map along the missing axis with a 1D Gaussian kernel.
  axial    : along D (z) axis
  coronal  : along H (y) axis
  sagittal : along W (x) axis
- Concatenate the three volumes along channel -> [B,3,P,P,P], then 3D conv fusion.

Also provides:
- gaussian_1d(P, sigma_ratio)
- embed_plane_to_volume(plane2d, axis, sigma_ratio)
- embed_three(y_ax, y_cor, y_sag, sigma_ratio)
- tri_consistency_loss(vol_ax, vol_cor, vol_sag, mask, mode)

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
from typing import Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_1d(P: int, sigma_ratio: float = 0.125, device=None, dtype=None) -> torch.Tensor:
    """
    Create a length-P 1D Gaussian (centered in the window), normalized to sum=1.
    sigma = sigma_ratio * P  (default: P/8)
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    sigma = max(1.0, sigma_ratio * float(P))
    coords = torch.arange(P, device=device, dtype=dtype) - (P - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / (g.sum() + 1e-6)
    return g  # [P]


def embed_plane_to_volume(
    plane2d: torch.Tensor,
    axis: Literal["axial", "coronal", "sagittal"] = "axial",
    sigma_ratio: float = 0.125,
) -> torch.Tensor:
    """
    plane2d: [B,1,P,P]  (2D prediction from a branch)
    returns: [B,1,P,P,P]  (embedded pseudo-volume)

    Axis meanings (output volume order [D,H,W]):
      - axial    : broadcast plane along D (z) -> kernel shaped [1,1,P,1,1]
      - coronal  : broadcast plane along H (y) -> kernel shaped [1,1,1,P,1]
      - sagittal : broadcast plane along W (x) -> kernel shaped [1,1,1,1,P]
    """
    B, C, P, Q = plane2d.shape
    assert C == 1 and P == Q, "Expect plane2d shape [B,1,P,P] with square size."
    dev, dt = plane2d.device, plane2d.dtype
    g = gaussian_1d(P, sigma_ratio=sigma_ratio, device=dev, dtype=dt)

    if axis == "axial":
        # [B,1,1,P,P] * [1,1,P,1,1] -> [B,1,P,P,P]
        vol = plane2d.unsqueeze(2) * g.view(1, 1, P, 1, 1)
    elif axis == "coronal":
        # plane is logically [D,W]; broadcast along H
        # [B,1,P,1,P] * [1,1,1,P,1] -> [B,1,P,P,P]
        vol = plane2d.unsqueeze(3) * g.view(1, 1, 1, P, 1)
    elif axis == "sagittal":
        # plane is logically [D,H]; broadcast along W
        # [B,1,P,P,1] * [1,1,1,1,P] -> [B,1,P,P,P]
        vol = plane2d.unsqueeze(4) * g.view(1, 1, 1, 1, P)
    else:
        raise ValueError(f"Unknown axis: {axis}")
    return vol


def embed_three(
    y_ax: torch.Tensor, y_cor: torch.Tensor, y_sag: torch.Tensor, sigma_ratio: float = 0.125
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    y_ax/y_cor/y_sag: [B,1,P,P]
    returns (vol_ax, vol_cor, vol_sag), each [B,1,P,P,P]
    """
    vol_ax  = embed_plane_to_volume(y_ax,  axis="axial",    sigma_ratio=sigma_ratio)
    vol_cor = embed_plane_to_volume(y_cor, axis="coronal",  sigma_ratio=sigma_ratio)
    vol_sag = embed_plane_to_volume(y_sag, axis="sagittal", sigma_ratio=sigma_ratio)
    return vol_ax, vol_cor, vol_sag


def tri_consistency_loss(
    vol_ax: torch.Tensor, vol_cor: torch.Tensor, vol_sag: torch.Tensor,
    mask: torch.Tensor | None = None, mode: Literal["l1", "gradl1"] = "l1"
) -> torch.Tensor:
    """
    Pairwise consistency among three embedded volumes (before fusion).
    Each volume: [B,1,P,P,P]; mask: [B,1,P,P,P] or broadcastable.
    """
    vols = (vol_ax, vol_cor, vol_sag)
    pairs = ((0, 1), (0, 2), (1, 2))
    loss = 0.0
    for i, j in pairs:
        a, b = vols[i], vols[j]
        if mode == "l1":
            if mask is None:
                loss = loss + (a - b).abs().mean()
            else:
                loss = loss + ((a - b).abs() * mask).sum() / (mask.sum() + 1e-6)
        elif mode == "gradl1":
            def _fd(t):
                dx = t[..., 1:, :, :] - t[..., :-1, :, :]
                dy = t[..., :, 1:, :] - t[..., :, :-1, :]
                dz = t[..., :, :, 1:] - t[..., :, :, :-1]
                return dx, dy, dz
            if mask is None:
                loss = loss + sum((x - y).abs().mean() for x, y in zip(_fd(a), _fd(b))) / 3.0
            else:
                m = mask[..., 1:, 1:, 1:]  # roughly align inner region for grads
                grads = [(x - y).abs().sum() / (m.sum() + 1e-6) for x, y in zip(_fd(a), _fd(b))]
                loss = loss + sum(grads) / 3.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return loss / 3.0


class Fusion3D(nn.Module):
    """
    3D fusion head over concatenated tri-plane embedded volumes.
    Input:  vol_3ch [B,3,P,P,P]
    Output: out     [B,1,P,P,P]
    """
    def __init__(self, in_ch: int = 3, base: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1),
            nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv3d(base, base, 3, padding=1),
            nn.GroupNorm(4, base), nn.SiLU(),
            nn.Conv3d(base, 1, 1)
        )

    def forward(self, vol_3ch: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        vol_3ch: [B,3,P,P,P]; mask: [B,1,P,P,P] or None
        """
        out = self.conv(vol_3ch)
        if mask is not None:
            out = out * mask
        return out
