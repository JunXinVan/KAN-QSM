# -*- coding: utf-8 -*-
"""
Calcification-Streak metric (engineering replica, stable v1.0)

本指标以“工程可复现”的方式近似 QSM Challenge 2.0 的 Calcification Streak 口径：
- 输入：预测 pred、真值 gt、脑掩膜 mask、钙化 ROI calc（四者同形状 [D,H,W]）
- 误差：e = pred - gt
- 高通：对 e 进行 3D LoG(σ) 过滤后取绝对值 |LoG(e)|
- 环形区：以 calc 形态学膨胀 (r_outer) 减去 (r_inner) 得到环形评估区 ring
- 对照区：brain∧(非 outer) 的区域（即环外脑内区域）作为对照
- 输出：
  * CSI_mean       = mean(|LoG(e)| @ ring)
  * CSI_norm_ratio = CSI_mean / (mean(|LoG(e)| @ control) + eps)
  * voxel 计数与若干辅助统计
  * 可选：以钙化质心为圆心的径向能量剖面（半径步长=1voxel）

注意：
- 这是“工程复刻”而非挑战官方实现；若需完全一致口径，应替换为官方评测脚本。
- 代码不依赖 SciPy，三维形态学膨胀使用 PyTorch 的 max_pool3d 实现。

用法：
  python metrics/calc_streak.py \
      --pred /path/pred.npy --gt /path/gt.npy \
      --mask /path/brain_mask.npy --calc /path/calc_roi.npy \
      --out_csv streak_report.csv \
      --sigma 1.0 --r_inner 2 --r_outer 8 --profile_csv radial_profile.csv
"""
from __future__ import annotations
import os, argparse, csv
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------
# IO helpers
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


# --------------------------
# 3D LoG 高通
# --------------------------
def _log_kernel3d(size: int, sigma: float, device, dtype) -> torch.Tensor:
    if size % 2 == 0:
        size += 1
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
    r2 = x * x + y * y + z * z
    sigma2 = sigma * sigma
    norm = -1.0 / (np.pi * sigma2 ** 2)
    k = norm * (1 - r2 / (2 * sigma2)) * torch.exp(-r2 / (2 * sigma2))
    k = k - k.mean()
    return k.view(1, 1, size, size, size)


def _hpf_log_abs(e: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """|LoG(e)|, e:[1,1,D,H,W] -> same shape"""
    ksize = int(round(6 * sigma))
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2
    k = _log_kernel3d(ksize, sigma, e.device, e.dtype)
    e_pad = F.pad(e, (pad, pad, pad, pad, pad, pad), mode='reflect')
    y = F.conv3d(e_pad, k)
    return y.abs()


# --------------------------
# 形态学膨胀（Chebyshev 距离）
# --------------------------
def _dilate(mask01: torch.Tensor, r: int) -> torch.Tensor:
    """mask01:[1,1,D,H,W] in {0,1} -> dilation radius r (max_pool3d)"""
    if r <= 0:
        return mask01
    k = 2 * r + 1
    # padding=r 保持尺寸不变；kernel 全 1 的 max_pool 等价于二值膨胀的结构元素为立方体
    out = F.max_pool3d(mask01, kernel_size=k, stride=1, padding=r)
    return (out > 0.5).float()


# --------------------------
# 主指标计算
# --------------------------
def compute_csi(
    pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, calc: np.ndarray,
    sigma: float = 1.0, r_inner: int = 2, r_outer: int = 8
) -> Dict[str, float]:
    if not (pred.shape == gt.shape == mask.shape == calc.shape):
        raise ValueError("pred/gt/mask/calc must share the same shape [D,H,W].")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # to torch
    P = torch.from_numpy(pred).to(device).float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    G = torch.from_numpy(gt  ).to(device).float().unsqueeze(0).unsqueeze(0)
    M = torch.from_numpy(mask).to(device).float().unsqueeze(0).unsqueeze(0)
    C = torch.from_numpy(calc).to(device).float().unsqueeze(0).unsqueeze(0)
    # error high-pass
    E = _hpf_log_abs(P - G, sigma=sigma) * M
    # rings
    C = (C > 0.5).float()
    outer = _dilate(C, r_outer)
    inner = _dilate(C, r_inner)
    ring = (outer - inner).clamp(min=0.0, max=1.0) * M
    control = (M - outer).clamp(min=0.0, max=1.0)  # 脑内非 outer 的区域
    # stats
    n_ring    = float(ring.sum().item())
    n_control = float(control.sum().item())
    if n_ring < 1.0:
        raise ValueError("Ring region is empty. Check r_inner/r_outer or calc mask.")
    # 平均能量
    m_ring    = float((E * ring).sum().item()) / (n_ring + 1e-6)
    m_control = float((E * control).sum().item()) / (max(n_control, 1.0))
    csi_ratio = m_ring / (m_control + 1e-8)
    # 总能量（可参考）
    s_ring    = float((E * ring).sum().item())
    s_control = float((E * control).sum().item())
    return dict(
        CSI_mean=m_ring,
        CSI_norm_ratio=csi_ratio,
        E_sum_ring=s_ring,
        E_sum_control=s_control,
        Vox_ring=n_ring,
        Vox_control=n_control
    )


# --------------------------
# 径向剖面（以质心为圆心）
# --------------------------
def radial_profile(
    vol_energy: np.ndarray, calc_mask: np.ndarray, mask: Optional[np.ndarray],
    r_max: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    以 calc 质心为圆心，统计半径 r=0..r_max 的平均能量（限定在脑内）。
    返回：r（整数半径），mean_energy[r]
    """
    D, H, W = vol_energy.shape
    # 质心
    idx = np.argwhere(calc_mask > 0.5)
    if idx.size == 0:
        raise ValueError("calc mask is empty.")
    cz, cy, cx = idx.mean(axis=0)  # 浮点质心
    # 网格
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    rr = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr.astype(np.float32)
    if mask is not None:
        valid = (mask > 0.5)
    else:
        valid = np.ones_like(vol_energy, dtype=bool)
    # 统计
    r_vals = np.arange(0, r_max + 1, dtype=np.int32)
    means = np.zeros_like(r_vals, dtype=np.float32)
    for i, r in enumerate(r_vals):
        shell = (rr >= r) & (rr < r + 1.0) & valid
        if shell.any():
            means[i] = vol_energy[shell].mean(dtype=np.float64)
        else:
            means[i] = np.nan
    return r_vals, means


# --------------------------
# CLI
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, required=True)
    ap.add_argument("--gt",   type=str, required=True)
    ap.add_argument("--mask", type=str, required=True)
    ap.add_argument("--calc", type=str, required=True)
    ap.add_argument("--sigma", type=float, default=1.0, help="LoG sigma")
    ap.add_argument("--r_inner", type=int, default=2, help="inner dilation radius (voxels)")
    ap.add_argument("--r_outer", type=int, default=8, help="outer dilation radius (voxels)")
    ap.add_argument("--profile_csv", type=str, default=None, help="Optional output radial profile CSV")
    ap.add_argument("--out_csv", type=str, required=True, help="Summary CSV path")
    args = ap.parse_args()

    pred = _load_vol(args.pred)
    gt   = _load_vol(args.gt)
    mask = _load_vol(args.mask)
    calc = _load_vol(args.calc)

    # 主指标
    stats = compute_csi(pred, gt, mask, calc, sigma=args.sigma,
                        r_inner=args.r_inner, r_outer=args.r_outer)

    # 径向剖面（基于 |LoG(e)|）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = torch.from_numpy(pred - gt).to(device).float().unsqueeze(0).unsqueeze(0)
    E = _hpf_log_abs(E, sigma=args.sigma).squeeze().detach().cpu().numpy()
    r_vals, means = radial_profile(E, calc, mask, r_max=max(args.r_outer * 2, 20))

    # 写 summary CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stats.items():
            w.writerow([k, f"{float(v):.8f}"])
    print(f"[OK] Summary saved to {args.out_csv}")

    # 写剖面 CSV（可选）
    if args.profile_csv:
        with open(args.profile_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["radius_voxel", "mean_absLoG_error"])
            for r, m in zip(r_vals, means):
                w.writerow([int(r), ("" if np.isnan(m) else f"{float(m):.8f}")])
        print(f"[OK] Radial profile saved to {args.profile_csv}")


if __name__ == "__main__":
    main()
