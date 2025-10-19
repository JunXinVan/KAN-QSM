# -*- coding: utf-8 -*-
"""
Visualization tools (stable v1.0)

Subcommands
-----------
1) spectrum
   生成体数据的 k-space 热图（中心切片），支持 log 显示与可选 fftshift。
   用法：
     python tools/vis.py spectrum --vol /path/chi.npy --plane axial \
         --out heatmap.png --log --shift

2) csar-weights
   从训练 checkpoint 解析 CSAR 切片注意（Simple2DBranch: ax.csar.slice.*，UKAN2DBranch: ax.slice_fuser.*），
   计算 L 维权重曲线并导出 PNG + CSV。
   用法：
     python tools/vis.py csar-weights --ckpt /path/ckpt.pth --L 11 --T 1.0 \
         --out_png weights.png --out_csv weights.csv
"""
from __future__ import annotations
import os, argparse, csv
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------
# Utilities
# --------------------------
def _load_vol(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            key = "data" if "data" in z.files else z.files[0]
            arr = z[key]
    else:
        raise ValueError(f"Unsupported extension: {path}")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for this command. Please install it.") from e


# --------------------------
# 1) Spectrum heatmap
# --------------------------
def spectrum_heatmap(vol: np.ndarray, plane: str = "axial", log: bool = True, shift: bool = False) -> np.ndarray:
    """
    返回 2D 热图数组（float32）。
    - vol: [D,H,W]
    - plane in {'axial','coronal','sagittal'}
    - log: 是否使用 log1p(|F|)
    - shift: 是否 fftshift 以中心化 DC
    """
    V = torch.from_numpy(vol).float()
    Fk = torch.fft.fftn(V, dim=(0, 1, 2))
    if shift:
        Fk = torch.fft.fftshift(Fk, dim=(0, 1, 2))
    mag = torch.sqrt(Fk.real ** 2 + Fk.imag ** 2).cpu().numpy()
    D, H, W = mag.shape
    if plane == "axial":
        sl = mag[D // 2, :, :]
    elif plane == "coronal":
        sl = mag[:, H // 2, :]
    elif plane == "sagittal":
        sl = mag[:, :, W // 2]
    else:
        raise ValueError("plane must be axial|coronal|sagittal")
    sl = sl.astype(np.float32)
    if log:
        sl = np.log1p(sl)
    # 归一化到 [0,1]
    sl = sl - sl.min()
    mx = sl.max()
    if mx > 0:
        sl = sl / mx
    return sl


def cmd_spectrum(args):
    plt = _ensure_matplotlib()
    vol = _load_vol(args.vol)
    img = spectrum_heatmap(vol, plane=args.plane, log=args.log, shift=args.shift)
    h, w = img.shape
    fig = plt.figure(figsize=(max(3, w/128), max(3, h/128)), dpi=128)
    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=args.cmap, origin="lower")
    ax.set_title(f"k-space heatmap ({args.plane})")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("normalized magnitude (log1p)" if args.log else "normalized magnitude", rotation=90)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved spectrum heatmap to {args.out}")


# --------------------------
# 2) CSAR slice-attention weights
# --------------------------
def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-x.abs())) + x.clamp_min(0)


def extract_slice_params_from_ckpt(ckpt_path: str) -> Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    从 checkpoint 中自动解析 (variant, mu, P, D)
    variant: 'simple' -> keys 包含 'csar.slice.*'
             'ukan'   -> keys 包含 'slice_fuser.*'
    返回：variant, mu[1], P[L,rank], D[L]
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    keys = list(state.keys())

    # 优先 Simple2DBranch
    mu_k = [k for k in keys if k.endswith("csar.slice.mu")]
    P_k  = [k for k in keys if k.endswith("csar.slice.P")]
    D_k  = [k for k in keys if k.endswith("csar.slice.D")]
    variant = "simple"
    if not (mu_k and P_k and D_k):
        # 尝试 UKAN2DBranch
        mu_k = [k for k in keys if k.endswith("slice_fuser.mu")]
        P_k  = [k for k in keys if k.endswith("slice_fuser.P")]
        D_k  = [k for k in keys if k.endswith("slice_fuser.D")]
        variant = "ukan"
    if not (mu_k and P_k and D_k):
        raise RuntimeError("Cannot find slice-attention parameters in checkpoint.")

    mu = state[mu_k[0]].detach().clone().float().view(1)       # scalar
    P  = state[P_k[0]].detach().clone().float()                # [L,rank]
    D  = state[D_k[0]].detach().clone().float().view(-1)       # [L]
    return variant, mu, P, D


def csar_weights_from_params(L: int, mu: torch.Tensor, P: torch.Tensor, D: torch.Tensor, T: float) -> np.ndarray:
    """
    根据 (mu,P,D,T) 计算归一化权重 w ∈ R^L：
      var_i = sum_j P[i,j]^2 + softplus(D[i]) + eps
      w_i ∝ exp( -0.5 * ((i - mu)^2) / (var_i * T) )
    """
    device = mu.device
    idx = torch.arange(L, device=device, dtype=mu.dtype)
    var = (P ** 2).sum(dim=1) + _softplus(D) + 1e-5
    w = torch.exp(-0.5 * ((idx - mu) ** 2) / (var * float(T)))
    w = w / (w.sum() + 1e-6)
    return w.detach().cpu().numpy().astype(np.float32)


def cmd_csar_weights(args):
    plt = _ensure_matplotlib()
    variant, mu, P, D = extract_slice_params_from_ckpt(args.ckpt)
    L = int(args.L)
    w = csar_weights_from_params(L, mu, P, D, T=float(args.T))

    # Plot
    fig = plt.figure(figsize=(6, 3), dpi=128)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(L), w, marker="o")
    ax.set_title(f"Slice weights ({variant}, T={float(args.T):.2f})")
    ax.set_xlabel("slice index (0..L-1)"); ax.set_ylabel("weight")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved weights plot to {args.out_png}")

    # CSV
    with open(args.out_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["slice_index", "weight"])
        for i, wi in enumerate(w.tolist()):
            wtr.writerow([i, f"{wi:.8f}"])
    print(f"[OK] Saved weights CSV to {args.out_csv}")


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualization tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # spectrum
    sp = sub.add_parser("spectrum", help="Generate k-space heatmap")
    sp.add_argument("--vol", type=str, required=True, help="3D volume (.npy/.npz)")
    sp.add_argument("--plane", type=str, choices=["axial", "coronal", "sagittal"], default="axial")
    sp.add_argument("--out", type=str, required=True, help="Output PNG path")
    sp.add_argument("--log", action="store_true", help="Use log1p magnitude")
    sp.add_argument("--shift", action="store_true", help="fftshift for visualization")
    sp.add_argument("--cmap", type=str, default="magma")
    sp.set_defaults(func=cmd_spectrum)

    # csar-weights
    wp = sub.add_parser("csar-weights", help="Plot/export slice attention weights from checkpoint")
    wp.add_argument("--ckpt", type=str, required=True, help="checkpoint .pth saved by train.py/train_ukan.py")
    wp.add_argument("--L", type=int, required=True, help="number of stacked slices (must match training)")
    wp.add_argument("--T", type=float, default=1.0, help="temperature (TTA-lite)")
    wp.add_argument("--out_png", type=str, required=True)
    wp.add_argument("--out_csv", type=str, required=True)
    wp.set_defaults(func=cmd_csar_weights)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
