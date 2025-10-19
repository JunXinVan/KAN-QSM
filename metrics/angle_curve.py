# -*- coding: utf-8 -*-
"""
Angle-energy curve & streak index around the dipole "magic angle" (stable v1.0)

This script computes the energy distribution in k-space as a function of angle theta
w.r.t. B0 (assumed || z-axis), using the unshifted 3D FFT (full complex spectrum).
It can operate on:
  1) a single volume (e.g., susceptibility map) -> spectrum energy curve, or
  2) error volume (pred - gt) if both pred and gt are provided -> error spectrum curve.

We also compute a simple "streak index" (SI) measuring the fraction of energy
concentrated near the dipole zero cone (|cos(theta)| ≈ 1/sqrt(3) ≈ 0.577).
SI = Energy_in_band(|cosθ|∈[c0-w, c0+w] & r∈[rmin,rmax]) / Energy_total(r∈[rmin,rmax])

Usage:
  # On a single volume:
  python metrics/angle_curve.py --vol /path/chi.npy --out_csv chi_angle_curve.csv

  # On error volume (pred-gt):
  python metrics/angle_curve.py --pred /path/pred.npy --gt /path/gt.npy --out_csv err_angle_curve.csv

  # Optional parameters:
  --nbins 36 --rmin 0.05 --rmax 0.35 --band 0.05

Outputs:
  - CSV with columns: bin_center_cos, energy, energy_norm
  - Prints SI (streak index) to console

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
import os, argparse, csv
from typing import Tuple, Optional
import numpy as np
import torch


def load_volume(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            key = "data" if "data" in z.files else z.files[0]
            arr = z[key]
    else:
        raise ValueError(f"Unsupported file extension for {path}. Use .npy or .npz.")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def compute_angle_curve(
    vol: np.ndarray,
    nbins: int = 36,
    rmin: float = 0.05,
    rmax: float = 0.35,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute angle-energy curve using |FFT(vol)|^2 integrated over radial band and angle bins of |cosθ|.

    Returns:
      bin_centers_cos : [nbins], in [0,1]
      energy          : [nbins], unnormalized energy per bin
      total_energy    : scalar, total energy within the radial band
    """
    vol_t = torch.from_numpy(vol).to(device).float()
    D, H, W = vol_t.shape

    # Full FFT (unshifted)
    F = torch.fft.fftn(vol_t, dim=(0, 1, 2))
    power = (F.real ** 2 + F.imag ** 2)  # |F|^2

    # Frequency grid (unshifted)
    kz = torch.fft.fftfreq(D, d=1.0, device=device).view(D, 1, 1)
    ky = torch.fft.fftfreq(H, d=1.0, device=device).view(1, H, 1)
    kx = torch.fft.fftfreq(W, d=1.0, device=device).view(1, 1, W)
    r = torch.sqrt(kx * kx + ky * ky + kz * kz)  # radius (normalized to Nyquist=0.5)
    eps = 1e-12
    cos_th = torch.abs(kz) / (r + eps)           # |cos(theta)| w.r.t. z-axis (B0 direction)

    # Radial band mask (exclude DC and very high freqs)
    band = (r >= rmin) & (r <= rmax)

    # Bin edges for |cosθ| in [0,1]
    edges = torch.linspace(0.0, 1.0, nbins + 1, device=device)
    energy = torch.zeros(nbins, device=device, dtype=torch.float32)

    # Digitize cos_th into bins (vectorized)
    # Note: torch.bucketize returns indices in [1..len(edges)], right=False means (edges[i-1], edges[i]] intervals.
    idx = torch.bucketize(cos_th[band], edges, right=True) - 1
    idx = idx.clamp(0, nbins - 1)

    # Accumulate energy per bin
    flat_idx = idx.view(-1).long()
    flat_pow = power[band].view(-1)
    energy.scatter_add_(0, flat_idx, flat_pow)

    total_energy = float(energy.sum().item())
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.detach().cpu().numpy(), energy.detach().cpu().numpy(), total_energy


def streak_index_from_curve(
    centers_cos: np.ndarray, energy: np.ndarray, band_halfwidth: float = 0.05
) -> float:
    """
    Compute streak index near the dipole zero-cone: |cosθ| = 1/sqrt(3) ~ 0.577.
    SI = energy_in_band / total_energy.
    """
    c0 = 1.0 / np.sqrt(3.0)  # ~0.577
    in_band = (centers_cos >= max(0.0, c0 - band_halfwidth)) & (centers_cos <= min(1.0, c0 + band_halfwidth))
    num = float(energy[in_band].sum())
    den = float(energy.sum() + 1e-12)
    return num / den


def save_curve_csv(path_csv: str, centers_cos: np.ndarray, energy: np.ndarray) -> None:
    energy_norm = energy / (energy.sum() + 1e-12)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_center_cos", "energy", "energy_norm"])
        for c, e, en in zip(centers_cos, energy, energy_norm):
            w.writerow([f"{float(c):.6f}", f"{float(e):.6e}", f"{float(en):.6e}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", type=str, default=None, help="Single volume (.npy/.npz)")
    ap.add_argument("--pred", type=str, default=None, help="Prediction volume (.npy/.npz)")
    ap.add_argument("--gt",   type=str, default=None, help="Ground-truth volume (.npy/.npz)")
    ap.add_argument("--nbins", type=int, default=36)
    ap.add_argument("--rmin", type=float, default=0.05)
    ap.add_argument("--rmax", type=float, default=0.35)
    ap.add_argument("--band", type=float, default=0.05, help="Half-width around |cosθ|=1/sqrt(3) for SI")
    ap.add_argument("--out_csv", type=str, required=True, help="Output CSV path for curve")
    args = ap.parse_args()

    if args.vol is None and (args.pred is None or args.gt is None):
        raise ValueError("Provide either --vol, or both --pred and --gt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.vol is not None:
        vol = load_volume(args.vol)
    else:
        pred = load_volume(args.pred)
        gt   = load_volume(args.gt)
        if pred.shape != gt.shape:
            raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")
        vol = (pred - gt).astype(np.float32)

    centers, energy, total = compute_angle_curve(
        vol, nbins=args.nbins, rmin=args.rmin, rmax=args.rmax, device=device
    )
    si = streak_index_from_curve(centers, energy, band_halfwidth=args.band)
    save_curve_csv(args.out_csv, centers, energy)
    print(f"[OK] Saved angle-energy curve to {args.out_csv}")
    print(f"[Info] Streak Index (|cosθ|≈1/√3 ± {args.band:.3f}): {si:.6f}")

if __name__ == "__main__":
    main()
