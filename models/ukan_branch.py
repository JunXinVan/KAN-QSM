# -*- coding: utf-8 -*-
"""
UKAN 2D Branch wrapper (stable v1.0)
- Input : [B, L, 1, P, P]  (L stacked slices for one plane)
- Output: [B, 1, P, P]     (fused 2D prediction for that plane)

Design:
1) Slice fuse: SliceAttnLowRank (expectation only, no sampling) to fuse L slices -> [B,1,P,P]
2) Pre-projection 1x1 conv: 1ch -> 3ch  (robust to UKAN implementations that hardcode 3 input channels)
3) UKAN forward: returns [B, 1, P, P]  (num_classes=1)

Robustness:
- If UKAN arch cannot be imported from models/external/ukan, falls back to a light Conv block branch
  so that training/eval can still run.

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse our stable slice attention (no sampling)
from .csar_lite import SliceAttnLowRank

# --- Try to import official UKAN implementation from your local checkout ---
_UKAN_AVAILABLE = False
_UKAN_CLS = None
try:
    # Recommended placement: models/external/ukan/archs.py
    from .external.ukan.archs import UKAN as _UKAN_IMPL
    _UKAN_AVAILABLE = True
    _UKAN_CLS = _UKAN_IMPL
except Exception:
    try:
        # Alternative layout if you kept original folder name like "Seg_UKAN/archs.py"
        from .external.ukan.Seg_UKAN.archs import UKAN as _UKAN_IMPL2
        _UKAN_AVAILABLE = True
        _UKAN_CLS = _UKAN_IMPL2
    except Exception:
        _UKAN_AVAILABLE = False
        _UKAN_CLS = None


class _FallbackLight2D(nn.Module):
    """
    Fallback 2D branch if UKAN is not available.
    Keeps the interface identical (input [B,1,P,P] -> output [B,1,P,P]).
    """
    def __init__(self, base: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UKAN2DBranch(nn.Module):
    """
    UKAN 2D branch wrapper with slice-fuse front-end.
    Args:
      L           : number of stacked slices (odd)
      T           : temperature for SliceAttnLowRank (TTA-lite compatible)
      rank        : low-rank width in SliceAttnLowRank
      ukan_kwargs : dict passed to UKAN constructor (if available)
                    Example sensible defaults (works for many configs):
                        num_classes=1, in_chans=3, input_channels=3,
                        img_size=96, patch_size=3, embed_dims=[64,96,160], depths=[1,1,1]
      force_rgb   : if True, always pre-project 1ch->3ch before UKAN
    """
    def __init__(
        self,
        L: int = 11,
        T: float = 1.0,
        rank: int = 4,
        ukan_kwargs: Optional[dict] = None,
        force_rgb: bool = True,
        fallback_base: int = 64,
    ) -> None:
        super().__init__()
        self.L = int(L)
        self.slice_fuser = SliceAttnLowRank(L=L, rank=rank, T=T)
        self.force_rgb = bool(force_rgb)

        if _UKAN_AVAILABLE and _UKAN_CLS is not None:
            # Provide safe defaults; user can override via ukan_kwargs.
            if ukan_kwargs is None:
                ukan_kwargs = dict(
                    num_classes=1, in_chans=3, input_channels=3,
                    img_size=96, patch_size=3, embed_dims=[64, 96, 160], depths=[1, 1, 1]
                )
            self.pre_rgb = nn.Conv2d(1, 3, kernel_size=1) if self.force_rgb or (ukan_kwargs.get("in_chans", 3) == 3) else nn.Identity()
            # Instantiate official UKAN
            self.ukan = _UKAN_CLS(**ukan_kwargs)
            self._using_ukan = True
        else:
            # Fallback stable conv branch
            self.pre_rgb = nn.Identity()
            self.ukan = _FallbackLight2D(base=fallback_base)
            self._using_ukan = False

    @torch.no_grad()
    def set_slice_temperature(self, T: float) -> None:
        self.slice_fuser.T = float(T)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs: [B, L, 1, P, P]  -> fuse along L -> [B,1,P,P] -> (1->3) -> UKAN -> [B,1,P,P]
        """
        B, L, C, H, W = xs.shape
        assert L == self.L and C == 1, f"UKAN2DBranch expects [B,{self.L},1,P,P], got {xs.shape}"
        # Slice fuse on input images directly (no feature encoder): stable for regression
        fused = self.slice_fuser(xs.view(B, L, 1, H, W))  # [B,1,P,P]
        x2d = fused                                          # keep name clarity
        x2d = self.pre_rgb(x2d)                              # [B,1/3,P,P]
        y = self.ukan(x2d)                                   # [B,1,P,P] if UKAN num_classes=1
        return y
# -*- coding: utf-8 -*-
"""
Evaluation script for Tri-plane 2.5D QSM (stable v1.0)

Features:
- Loads a training checkpoint saved by train.py
- Builds a model that mirrors the checkpoint's TriPlaneModel structure
- Supports two branch types:
    * simple : Simple2DBranch (default; matches train.py)
    * ukan   : UKAN2DBranch (this repo, models/ukan_branch.py)
- Computes both patch-level metrics and stitched full-volume metrics
  (overlap-averaged using patch mask as weights)
- Metrics: nRMSE, HFEN(LoG), SSIM(3D), Freq-L1(rFFT)
- TTA-lite temperature calibration on validation split (optional)

Usage:
  python eval.py --ckpt checkpoints/epoch_0050_best.pth --config configs/default.yaml --branch simple
  python eval.py --ckpt checkpoints/epoch_0050_best.pth --branch ukan

If --config is omitted, the script will try to load cfg from checkpoint['cfg'].

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
import os, argparse, json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Local imports (reuse training dataset and metrics)
from data.datasets import TriPlaneQSM
from models.branch2d_ukan import Simple2DBranch
from models.ukan_branch import UKAN2DBranch
from models.triplane_fusion3d import (
    Fusion3D, embed_three, tri_consistency_loss
)
from losses.qsm_losses import QSMLoss, hfen_log3d, freq_l1_rfft3d, ssim3d


def ensure_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_nrmse(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is not None:
        if mask.ndim == 4: mask = mask.unsqueeze(1)
        pred = pred * mask; gt = gt * mask
    num = torch.sqrt(torch.clamp((pred - gt).pow(2).sum(), min=1e-12))
    den = torch.sqrt(torch.clamp((gt).pow(2).sum(), min=1e-12))
    return (num / den).item()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, float]:
    if mask is not None and mask.ndim == 4:
        mask = mask.unsqueeze(1)
    p_m = pred * mask if mask is not None else pred
    g_m = gt * mask if mask is not None else gt
    return dict(
        nRMSE=masked_nrmse(pred, gt, mask),
        HFEN=hfen_log3d(p_m, g_m).item(),
        SSIM=ssim3d(p_m, g_m).item(),
        FreqL1=freq_l1_rfft3d(pred, gt).item()
    )


class TriPlaneModelEval(nn.Module):
    """
    Mirror of TriPlaneModel used in train.py, with switchable branch class.
    """
    def __init__(self, L: int, branch: str = "simple", base: int = 64,
                 share_weights: bool = True, csar_rank: int = 4, csar_T: float = 1.0,
                 sem_hidden_ratio: float = 0.5) -> None:
        super().__init__()
        if branch == "ukan":
            Branch = lambda: UKAN2DBranch(L=L, T=csar_T, rank=csar_rank)
        else:
            Branch = lambda: Simple2DBranch(L=L, in_ch=1, base=base,
                                            csar_rank=csar_rank, csar_T=csar_T,
                                            sem_hidden_ratio=sem_hidden_ratio)
        if share_weights:
            self.ax = self.cor = self.sag = Branch()
        else:
            self.ax, self.cor, self.sag = Branch(), Branch(), Branch()
        self.fuse3d = Fusion3D(in_ch=3, base=32)

    @torch.no_grad()
    def set_slice_temperature(self, T: float) -> None:
        # both Simple2DBranch and UKAN2DBranch expose .csar.slice or .slice_fuser
        if hasattr(self.ax, "csar"):  # Simple2DBranch
            self.ax.csar.slice.T = float(T); self.cor.csar.slice.T = float(T); self.sag.csar.slice.T = float(T)
        if hasattr(self.ax, "slice_fuser"):  # UKAN2DBranch
            self.ax.slice_fuser.T = float(T); self.cor.slice_fuser.T = float(T); self.sag.slice_fuser.T = float(T)

    def forward(self, ax: torch.Tensor, cor: torch.Tensor, sag: torch.Tensor, mask3d: torch.Tensor):
        y_ax  = self.ax(ax)
        y_cor = self.cor(cor)
        y_sag = self.sag(sag)
        vol_ax, vol_cor, vol_sag = embed_three(y_ax, y_cor, y_sag, sigma_ratio=0.125)
        vol_3ch = torch.cat([vol_ax, vol_cor, vol_sag], dim=1)
        pred = self.fuse3d(vol_3ch, mask=mask3d)
        aux = dict(y_ax=y_ax, y_cor=y_cor, y_sag=y_sag,
                   vol_ax=vol_ax, vol_cor=vol_cor, vol_sag=vol_sag)
        return pred, aux


def load_cfg_from_ckpt_or_yaml(ckpt_path: str, yaml_path: Optional[str]) -> Dict[str, Any]:
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
            return ckpt["cfg"]
    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)
    raise ValueError("Neither checkpoint nor config YAML contains a valid cfg dict.")


def stitch_init_arrays(ds: TriPlaneQSM) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    preds_sum, weights_sum = [], []
    for v in ds.vols:
        D, H, W = v.shape
        preds_sum.append(np.zeros((D, H, W), dtype=np.float32))
        weights_sum.append(np.zeros((D, H, W), dtype=np.float32))
    return preds_sum, weights_sum


def add_patch_to_stitch(
    preds_sum: List[np.ndarray], weights_sum: List[np.ndarray],
    subject_index: int, z0: int, y0: int, x0: int,
    patch_pred: np.ndarray, patch_mask: np.ndarray
) -> None:
    """
    Place patch_pred [P,P,P] into the subject's accumulator at z0:y0:x0,
    adding weights from patch_mask [P,P,P].
    """
    P = patch_pred.shape[0]
    ps = slice(z0, z0 + P), slice(y0, y0 + P), slice(x0, x0 + P)
    preds_sum[subject_index][ps] += patch_pred
    weights_sum[subject_index][ps] += patch_mask


def finalize_stitched(preds_sum: List[np.ndarray], weights_sum: List[np.ndarray]) -> List[np.ndarray]:
    outs = []
    for ps, ws in zip(preds_sum, weights_sum):
        out = ps / (ws + 1e-6)
        outs.append(out.astype(np.float32))
    return outs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config (fallback)")
    parser.add_argument("--branch", type=str, choices=["simple", "ukan"], default="simple")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tta", action="store_true", help="Enable TTA-lite temperature calibration")
    parser.add_argument("--tta_candidates", type=float, nargs="*", default=[0.8, 1.0, 1.2])
    parser.add_argument("--stitch", action="store_true", help="Also compute stitched full-volume metrics")
    args = parser.parse_args()

    device = ensure_device()
    print(f"[Eval] device = {device}")

    # Load cfg (from ckpt if present; else YAML)
    cfg = load_cfg_from_ckpt_or_yaml(args.ckpt, args.config)
    # Data
    if "data" not in cfg or "val" not in cfg["data"]:
        raise ValueError("Config must have data.val with vols/masks/chis.")
    Vva = [np.load(p).astype(np.float32) for p in cfg["data"]["val"]["vols"]]
    Mva = [np.load(p).astype(np.float32) for p in cfg["data"]["val"]["masks"]]
    Cva = [np.load(p).astype(np.float32) for p in cfg["data"]["val"]["chis"]]

    l = int(cfg.get("l", 11))
    patch = int(cfg.get("patch", 96))
    stride = int(cfg.get("stride", 64))
    normalize = cfg.get("normalize", "zscore")

    ds_val = TriPlaneQSM(Vva, Mva, Cva, l=l, patch=patch, stride=stride, normalize=normalize)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # Model hyperparams
    train_cfg = cfg.get("train", {})
    share_branch = bool(train_cfg.get("share_branch_weights", True))
    base = int(cfg.get("base", 64))
    csar_rank = int(cfg.get("csar_rank", 4))
    csar_T = float(cfg.get("csar_T", 1.0))
    sem_hidden_ratio = float(cfg.get("sem_hidden_ratio", 0.5))

    model = TriPlaneModelEval(
        L=l, branch=args.branch, base=base, share_weights=share_branch,
        csar_rank=csar_rank, csar_T=csar_T, sem_hidden_ratio=sem_hidden_ratio
    ).to(device)

    # Load checkpoint (state_dict under key "model")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "model" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing or unexpected:
            print(f"[Warn] Missing keys: {missing}\n[Warn] Unexpected keys: {unexpected}")
    else:
        # if checkpoint was saved differently, try load directly
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            print(f"[Warn] Missing keys: {missing}\n[Warn] Unexpected keys: {unexpected}")

    # Optional: TTA-lite temperature calibration
    if args.tta and args.tta_candidates:
        bestT, bestNRMSE = None, float("inf")
        for T in args.tta_candidates:
            model.set_slice_temperature(T)
            # patch-level val
            model.eval()
            sum_metrics = dict(nRMSE=0.0, HFEN=0.0, SSIM=0.0, FreqL1=0.0); total = 0
            with torch.no_grad():
                for batch in dl_val:
                    ax, cor, sag = batch["ax"].to(device).float(), batch["cor"].to(device).float(), batch["sag"].to(device).float()
                    mask, gt = batch["mask"].to(device).float(), batch["chi"].to(device).float()
                    pred, _ = model(ax, cor, sag, mask)
                    m = compute_metrics(pred, gt, mask)
                    for k in sum_metrics: sum_metrics[k] += m[k] * ax.size(0)
                    total += ax.size(0)
            avg = {k: v / total for k, v in sum_metrics.items()}
            if avg["nRMSE"] < bestNRMSE:
                bestNRMSE, bestT = avg["nRMSE"], float(T)
        model.set_slice_temperature(bestT)
        print(f"[TTA-lite] Best T = {bestT:.3f} (nRMSE={bestNRMSE:.6f})")

    # Final evaluation (patch-level)
    model.eval()
    patch_sum = dict(nRMSE=0.0, HFEN=0.0, SSIM=0.0, FreqL1=0.0); nP = 0

    # For stitching full volumes
    if args.stitch:
        preds_sum, weights_sum = stitch_init_arrays(ds_val)

    with torch.no_grad():
        for batch in dl_val:
            ax = batch["ax"].to(device).float()
            cor = batch["cor"].to(device).float()
            sag = batch["sag"].to(device).float()
            mask = batch["mask"].to(device).float()
            gt = batch["chi"].to(device).float()
            pred, _ = model(ax, cor, sag, mask)

            # Patch metrics
            pm = compute_metrics(pred, gt, mask)
            for k in patch_sum: patch_sum[k] += pm[k] * ax.size(0)
            nP += ax.size(0)

            # Stitch
            if args.stitch:
                # Convert to np for accumulator
                pred_np = pred.squeeze(1).cpu().numpy()     # [B,P,P,P]
                mask_np = mask.squeeze(1).cpu().numpy()     # [B,P,P,P]
                meta = batch["meta"].cpu().numpy()          # [B, 6] (si, z0, y0, x0, P, l)
                for b in range(pred_np.shape[0]):
                    si, z0, y0, x0, P, L = map(int, meta[b])
                    add_patch_to_stitch(preds_sum, weights_sum, si, z0, y0, x0, pred_np[b], mask_np[b])

    patch_avg = {k: v / max(1, nP) for k, v in patch_sum.items()}
    print(f"[Patch] nRMSE={patch_avg['nRMSE']:.6f} | HFEN={patch_avg['HFEN']:.6f} | "
          f"SSIM={patch_avg['SSIM']:.6f} | FreqL1={patch_avg['FreqL1']:.6f}")

    # Volume-level metrics
    if args.stitch:
        preds_vol = finalize_stitched(preds_sum, weights_sum)
        # Compute per-subject metrics
        nS = len(preds_vol)
        vol_sum = dict(nRMSE=0.0, HFEN=0.0, SSIM=0.0, FreqL1=0.0)
        for si in range(nS):
            pv = torch.from_numpy(preds_vol[si]).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
            gv = torch.from_numpy(ds_val.chis[si]).unsqueeze(0).unsqueeze(0).to(device)
            mv = torch.from_numpy(ds_val.masks[si]).unsqueeze(0).unsqueeze(0).to(device)
            mv = mv.float()
            mv = (mv > 0.5).float()
            m = compute_metrics(pv, gv, mv)
            for k in vol_sum: vol_sum[k] += m[k]
        vol_avg = {k: v / max(1, nS) for k, v in vol_sum.items()}
        print(f"[Volume] nRMSE={vol_avg['nRMSE']:.6f} | HFEN={vol_avg['HFEN']:.6f} | "
              f"SSIM={vol_avg['SSIM']:.6f} | FreqL1={vol_avg['FreqL1']:.6f}")


if __name__ == "__main__":
    main()
