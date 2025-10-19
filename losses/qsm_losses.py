# -*- coding: utf-8 -*-
"""
QSM training losses (stable v0.9)

Includes:
- L1
- 3D-SSIM (self-implemented with Gaussian 3D window)
- Gradient L1 (forward differences)
- HFEN via 3D LoG convolution
- Frequency L1 on rFFT half-spectrum
- Composite QSMLoss

All functions are torch differentiable and channel-agnostic (expect [B,1,D,H,W]).
Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Basic building blocks
# --------------------------
def l1_loss(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    if w is None:
        return (x - y).abs().mean()
    return ((x - y).abs() * w).sum() / (w.sum() + 1e-6)


def grad_l1_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def _fd(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = t[..., 1:, :, :] - t[..., :-1, :, :]
        dy = t[..., :, 1:, :] - t[..., :, :-1, :]
        dz = t[..., :, :, 1:] - t[..., :, :, :-1]
        return dx, dy, dz

    gx = _fd(x); gy = _fd(y)
    return sum((a - b).abs().mean() for a, b in zip(gx, gy)) / 3.0


# --------------------------
# 3D SSIM
# --------------------------
def _gaussian_kernel1d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _gaussian_kernel3d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    g = _gaussian_kernel1d(size, sigma, device, dtype)
    k = torch.einsum('i,j,k->ijk', g, g, g)  # [S,S,S]
    k = k / k.sum()
    k = k.view(1, 1, size, size, size)       # [1,1,S,S,S]
    return k


def ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Computes 3D SSIM over [B,1,D,H,W].
    Returns mean SSIM over batch.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch in SSIM: {x.shape} vs {y.shape}")
    if x.dim() != 5 or x.size(1) != 1:
        raise ValueError(f"SSIM expects [B,1,D,H,W], got {x.shape}")

    pad = window_size // 2
    w = _gaussian_kernel3d(window_size, sigma, x.device, x.dtype)

    # reflect padding to mimic 'same' conv
    def _conv(t: torch.Tensor) -> torch.Tensor:
        tpad = F.pad(t, (pad, pad, pad, pad, pad, pad), mode='reflect')
        return F.conv3d(tpad, w)

    mu_x = _conv(x)
    mu_y = _conv(y)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _conv(x * x) - mu_x2
    sigma_y2 = _conv(y * y) - mu_y2
    sigma_xy = _conv(x * y) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


# --------------------------
# HFEN via 3D LoG convolution
# --------------------------
def _log_kernel3d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    """
    Discrete 3D Laplacian-of-Gaussian (LoG) kernel.
    size should be odd; recommend ~ round(6*sigma)|odd.
    """
    if size % 2 == 0:
        size += 1
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    r2 = x * x + y * y + z * z
    sigma2 = sigma * sigma
    norm = -1.0 / (math.pi * sigma2 ** 2)
    log = norm * (1 - r2 / (2 * sigma2)) * torch.exp(-r2 / (2 * sigma2))
    # zero-mean normalization (sum close to 0)
    log = log - log.mean()
    log = log.view(1, 1, size, size, size)
    return log


def hfen_log3d(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    HFEN as L1 between LoG-filtered volumes.
    x,y: [B,1,D,H,W]
    """
    ksize = int(round(6 * sigma))
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    k = _log_kernel3d(ksize, sigma, x.device, x.dtype)

    def _f(t: torch.Tensor) -> torch.Tensor:
        tpad = F.pad(t, (pad, pad, pad, pad, pad, pad), mode='reflect')
        return F.conv3d(tpad, k)

    return (torch.abs(_f(x) - _f(y))).mean()


# --------------------------
# Frequency L1 on half-spectrum
# --------------------------
def freq_l1_rfft3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    L1 distance on rFFT half-spectrum (unshifted).
    x,y: [B,1,D,H,W]
    """
    X = torch.fft.rfftn(x, s=x.shape[-3:], dim=(-3, -2, -1))
    Y = torch.fft.rfftn(y, s=y.shape[-3:], dim=(-3, -2, -1))
    return torch.abs(X - Y).mean()


# --------------------------
# Composite QSM loss
# --------------------------
class QSMLoss(nn.Module):
    """
    Composite QSM loss:
    L = w_l1 * L1 + w_grad * GradL1 + w_hfen * HFEN(LoG) + w_freq * rFFT-L1 + w_ssim * (1-SSIM)
    SSIM weight default small; set to 0 if you prefer.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_grad: float = 0.05,
        w_hfen: float = 0.1,
        w_freq: float = 0.05,
        w_ssim: float = 0.05,
        ssim_window: int = 11,
        ssim_sigma: float = 1.5,
        hfen_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_grad = float(w_grad)
        self.w_hfen = float(w_hfen)
        self.w_freq = float(w_freq)
        self.w_ssim = float(w_ssim)
        self.ssim_window = int(ssim_window)
        self.ssim_sigma = float(ssim_sigma)
        self.hfen_sigma = float(hfen_sigma)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        pred, gt: [B,1,D,H,W]; mask: [B,1,D,H,W] or [1,D,H,W] or None
        """
        if pred.shape != gt.shape:
            raise ValueError(f"QSMLoss: pred/gt shape mismatch {pred.shape} vs {gt.shape}")
        if mask is not None:
            if mask.ndim == 4:
                mask = mask.unsqueeze(1)  # -> [B,1,D,H,W] if given as [B,D,H,W]
            if mask.shape != pred.shape:
                # Allow broadcast from [1,1,D,H,W]
                try:
                    mask = mask.expand_as(pred)
                except Exception:
                    raise ValueError(f"QSMLoss: mask shape {mask.shape} not broadcastable to pred {pred.shape}")

        L = 0.0
        L = L + self.w_l1 * l1_loss(pred, gt, mask)
        L = L + self.w_grad * grad_l1_loss(pred, gt)
        L = L + self.w_hfen * hfen_log3d(pred, gt, sigma=self.hfen_sigma)
        L = L + self.w_freq * freq_l1_rfft3d(pred, gt)
        if self.w_ssim > 0:
            ssim_val = ssim3d(pred, gt, window_size=self.ssim_window, sigma=self.ssim_sigma, data_range=1.0)
            L = L + self.w_ssim * (1.0 - ssim_val)
        return L
# Minimal, stable defaults for Tri-plane 2.5D training (v0.9)
l: 11          # number of stacked slices per plane (odd)
patch: 96      # 3D patch size (P)
stride: 64     # sliding-window stride
epochs: 50

# Data normalization: null | zscore | minmax
normalize: zscore

# Loss weights (QSMLoss)
loss:
  w_l1:   1.0
  w_grad: 0.05
  w_hfen: 0.10
  w_freq: 0.05
  w_ssim: 0.05
  ssim_window: 11
  ssim_sigma: 1.5
  hfen_sigma: 1.0
