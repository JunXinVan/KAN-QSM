# -*- coding: utf-8 -*-
"""
Train script for Tri-plane 2.5D QSM (stable v0.9)

Pipeline:
  - Load volumes/masks/chis (.npy/.npz) for train/val
  - Dataset: TriPlaneQSM -> [ax, cor, sag] stacks + [mask, chi]
  - Model: three 2D branches (shared or separate) + 3D fusion head
  - Loss: QSMLoss (+ optional Tri-consistency)
  - Val: metrics (nRMSE/HFEN/SSIM/FreqL1)
  - TTA-lite: calibrate slice-attention temperature T on val set from a candidate list
  - Save: periodic checkpoints + best checkpoint by nRMSE

Usage:
  python train.py --config configs/default.yaml

Config (YAML) MUST provide data paths:
  data:
    train:
      vols:  ["/path/train/vol_001.npy", ...]   # each [D,H,W], float32 or convertible
      masks: ["/path/train/mask_001.npy", ...]  # [D,H,W], (0..1)
      chis:  ["/path/train/chi_001.npy",  ...]  # [D,H,W]
    val:
      vols:  ["/path/val/vol_001.npy", ...]
      masks: ["/path/val/mask_001.npy", ...]
      chis:  ["/path/val/chi_001.npy",  ...]

Other keys (with defaults if missing):
  seed: 2025
  device: "cuda"
  out_dir: "checkpoints"
  l: 11
  patch: 96
  stride: 64
  normalize: "zscore"
  train:
    epochs: 50
    batch_size: 1
    lr: 2.0e-4
    weight_decay: 1.0e-4
    amp: true
    grad_accum: 1
    save_every: 5
    share_branch_weights: true
  tri_consistency:
    enabled: true
    weight: 0.02
    mode: "l1"   # or "gradl1"
    start_epoch:  int (default: epochs//2)
  tta_lite:
    enabled: true
    candidates: [0.8, 1.0, 1.2]

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
import os, sys, math, argparse, time, json, shutil, random
from typing import List, Tuple, Dict, Any, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback to plain prints

import yaml

# Local modules
from data.datasets import TriPlaneQSM
from models.branch2d_ukan import Simple2DBranch
from models.triplane_fusion3d import (
    Fusion3D,
    embed_three,
    tri_consistency_loss,
)
from losses.qsm_losses import QSMLoss, hfen_log3d, freq_l1_rfft3d, ssim3d


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_array(path: str) -> np.ndarray:
    """
    Load .npy or .npz array -> np.float32, shape [D,H,W]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            # common keys: 'arr_0' or 'data'
            key = "data" if "data" in z.files else z.files[0]
            arr = z[key]
    else:
        raise ValueError(f"Unsupported file extension for {path}, use .npy or .npz")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W] from {path}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_split_lists(cfg: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
                                                   List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Read train/val paths from cfg['data'] and load arrays."""
    if "data" not in cfg or "train" not in cfg["data"] or "val" not in cfg["data"]:
        raise ValueError("Config must define data.train and data.val with vols/masks/chis path lists.")
    def _read_split(split: Dict[str, Any]):
        vols = split.get("vols", [])
        masks = split.get("masks", [])
        chis = split.get("chis", [])
        if not (len(vols) == len(masks) == len(chis) and len(vols) > 0):
            raise ValueError("Each split must provide non-empty equal-length lists: vols/masks/chis.")
        V = [load_array(p) for p in vols]
        M = [load_array(p) for p in masks]
        C = [load_array(p) for p in chis]
        return V, M, C
    Vtr, Mtr, Ctr = _read_split(cfg["data"]["train"])
    Vva, Mva, Cva = _read_split(cfg["data"]["val"])
    return Vtr, Mtr, Ctr, Vva, Mva, Cva


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True).float()
        else:
            out[k] = v
    return out


def masked_nrmse(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    """
    nRMSE = ||(pred-gt) * mask||_2 / ||gt * mask||_2
    """
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)
        pred = pred * mask
        gt = gt * mask
    num = torch.sqrt(torch.clamp((pred - gt).pow(2).sum(), min=1e-12))
    den = torch.sqrt(torch.clamp((gt).pow(2).sum(), min=1e-12))
    return (num / den).item()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, float]:
    """
    pred, gt: [B,1,D,H,W], mask: [B,1,D,H,W] or [1,D,H,W] or None
    Returns averaged metrics over batch.
    """
    with torch.no_grad():
        if mask is not None:
            if mask.ndim == 4:
                mask = mask.unsqueeze(1)
            # for HFEN/SSIM we use masked volumes (simple multiplication)
            p_m = pred * mask
            g_m = gt * mask
        else:
            p_m, g_m = pred, gt

        nrmse = masked_nrmse(pred, gt, mask)
        hfen = hfen_log3d(p_m, g_m).item()
        ssim = ssim3d(p_m, g_m).item()
        fL1 = freq_l1_rfft3d(pred, gt).item()
        return dict(nRMSE=nrmse, HFEN=hfen, SSIM=ssim, FreqL1=fL1)


def log_metrics(tag: str, metrics: Dict[str, float]) -> str:
    return (f"{tag} | nRMSE {metrics['nRMSE']:.6f} | HFEN {metrics['HFEN']:.6f} | "
            f"SSIM {metrics['SSIM']:.6f} | FreqL1 {metrics['FreqL1']:.6f}")


# ------------------------------
# Model wrapper
# ------------------------------
class TriPlaneModel(nn.Module):
    """
    Three 2D branches (ax/cor/sag) + 3D fusion head.
    You can share branch weights (single instance) or use three separate instances.
    """
    def __init__(self, L: int = 11, base: int = 64, share_weights: bool = True,
                 csar_rank: int = 4, csar_T: float = 1.0, sem_hidden_ratio: float = 0.5) -> None:
        super().__init__()
        if share_weights:
            self.ax = self.cor = self.sag = Simple2DBranch(
                L=L, in_ch=1, base=base, csar_rank=csar_rank, csar_T=csar_T,
                sem_hidden_ratio=sem_hidden_ratio
            )
        else:
            self.ax  = Simple2DBranch(L=L, in_ch=1, base=base, csar_rank=csar_rank, csar_T=csar_T,
                                      sem_hidden_ratio=sem_hidden_ratio)
            self.cor = Simple2DBranch(L=L, in_ch=1, base=base, csar_rank=csar_rank, csar_T=csar_T,
                                      sem_hidden_ratio=sem_hidden_ratio)
            self.sag = Simple2DBranch(L=L, in_ch=1, base=base, csar_rank=csar_rank, csar_T=csar_T,
                                      sem_hidden_ratio=sem_hidden_ratio)
        self.fuse3d = Fusion3D(in_ch=3, base=32)

    @torch.no_grad()
    def set_slice_temperature(self, T: float) -> None:
        """
        Adjust CSAR-Lite slice-attention temperature (TTA-lite).
        Works for shared or separate branches.
        """
        # self.ax.csar.slice.T exists for Simple2DBranch
        self.ax.csar.slice.T = float(T)
        self.cor.csar.slice.T = float(T)
        self.sag.csar.slice.T = float(T)

    def forward(self, ax: torch.Tensor, cor: torch.Tensor, sag: torch.Tensor,
                mask3d: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        ax/cor/sag: [B,L,1,P,P]
        mask3d: [B,1,P,P,P]
        returns:
            pred3d: [B,1,P,P,P]
            aux   : dict('y_ax','y_cor','y_sag','vol_ax','vol_cor','vol_sag')
        """
        y_ax  = self.ax(ax)    # [B,1,P,P]
        y_cor = self.cor(cor)
        y_sag = self.sag(sag)

        vol_ax, vol_cor, vol_sag = embed_three(y_ax, y_cor, y_sag, sigma_ratio=0.125)
        vol_3ch = torch.cat([vol_ax, vol_cor, vol_sag], dim=1)  # [B,3,P,P,P]
        pred3d = self.fuse3d(vol_3ch, mask=mask3d)
        aux = dict(y_ax=y_ax, y_cor=y_cor, y_sag=y_sag,
                   vol_ax=vol_ax, vol_cor=vol_cor, vol_sag=vol_sag)
        return pred3d, aux


# ------------------------------
# Training / Validation
# ------------------------------
def train_one_epoch(
    model: TriPlaneModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_fn: QSMLoss,
    device: torch.device,
    use_amp: bool,
    tri_cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cnt = 0
    tri_weight = 0.0
    if tri_cfg.get("enabled", True):
        start_ep = tri_cfg.get("start_epoch", None)
        if start_ep is None:
            # default: start from 2nd half of training
            start_ep = max(1, cfg_train["epochs"] // 2)
        if epoch >= start_ep:
            tri_weight = float(tri_cfg.get("weight", 0.02))
        tri_mode = tri_cfg.get("mode", "l1")
    else:
        tri_mode = "l1"

    for batch in (tqdm(loader, desc=f"Train[{epoch}]") if tqdm else loader):
        batch = to_device(batch, device)
        ax, cor, sag = batch["ax"], batch["cor"], batch["sag"]     # [B,L,1,P,P]
        mask, gt = batch["mask"], batch["chi"]                      # [B,1,P,P,P]

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, aux = model(ax, cor, sag, mask)
            loss = loss_fn(pred, gt, mask)
            if tri_weight > 0.0:
                tri_loss = tri_consistency_loss(aux["vol_ax"], aux["vol_cor"], aux["vol_sag"],
                                                mask=mask, mode=tri_mode)
                loss = loss + tri_weight * tri_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = ax.size(0)
        total_loss += loss.item() * bs
        total_cnt += bs

    return {"loss": total_loss / max(1, total_cnt)}


@torch.no_grad()
def validate(
    model: TriPlaneModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    sum_metrics = dict(nRMSE=0.0, HFEN=0.0, SSIM=0.0, FreqL1=0.0)
    total = 0
    for batch in (tqdm(loader, desc="Val") if tqdm else loader):
        batch = to_device(batch, device)
        ax, cor, sag = batch["ax"], batch["cor"], batch["sag"]
        mask, gt = batch["mask"], batch["chi"]
        pred, _ = model(ax, cor, sag, mask)
        m = compute_metrics(pred, gt, mask)
        for k in sum_metrics:
            sum_metrics[k] += m[k] * ax.size(0)
        total += ax.size(0)
    if total == 0: total = 1
    return {k: v / total for k, v in sum_metrics.items()}


@torch.no_grad()
def calibrate_tta_temperature(
    model: TriPlaneModel,
    loader: DataLoader,
    device: torch.device,
    candidates: Sequence[float],
) -> float:
    """
    Scan T candidates and pick one with lowest nRMSE over the whole val set.
    """
    best_T, best_score = candidates[0], float("inf")
    for T in candidates:
        model.set_slice_temperature(T)
        metrics = validate(model, loader, device)
        if metrics["nRMSE"] < best_score:
            best_score = metrics["nRMSE"]
            best_T = T
    # set best T for subsequent test/eval
    model.set_slice_temperature(best_T)
    return float(best_T)


def save_checkpoint(
    out_dir: str,
    epoch: int,
    model: TriPlaneModel,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best: Dict[str, Any],
    cfg: Dict[str, Any],
    tag: str = "",
) -> None:
    ensure_dir(out_dir)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "best": best,
        "cfg": cfg,
    }
    fname = f"epoch_{epoch:04d}{('_'+tag) if tag else ''}.pth"
    torch.save(state, os.path.join(out_dir, fname))


# ------------------------------
# Main
# ------------------------------
def main(cfg_path: str) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Global defaults
    seed = int(cfg.get("seed", 2025))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = cfg.get("out_dir", "checkpoints")
    ensure_dir(out_dir)
    set_seed(seed)

    # Train settings
    global cfg_train
    cfg_train = cfg.get("train", {})
    epochs = int(cfg_train.get("epochs", cfg.get("epochs", 50)))
    batch_size = int(cfg_train.get("batch_size", 1))
    lr = float(cfg_train.get("lr", 2.0e-4))
    wd = float(cfg_train.get("weight_decay", 1.0e-4))
    use_amp = bool(cfg_train.get("amp", True))
    grad_accum = int(cfg_train.get("grad_accum", 1))
    save_every = int(cfg_train.get("save_every", 5))
    share_branch = bool(cfg_train.get("share_branch_weights", True))

    # Tri-consistency
    tri_cfg = cfg.get("tri_consistency", {})
    if "start_epoch" not in tri_cfg:
        tri_cfg["start_epoch"] = epochs // 2

    # TTA-lite
    tta_cfg = cfg.get("tta_lite", {"enabled": True, "candidates": [0.8, 1.0, 1.2]})
    tta_enabled = bool(tta_cfg.get("enabled", True))
    tta_candidates = tta_cfg.get("candidates", [0.8, 1.0, 1.2])

    # Data & Dataset
    Vtr, Mtr, Ctr, Vva, Mva, Cva = load_split_lists(cfg)
    l = int(cfg.get("l", 11))
    patch = int(cfg.get("patch", 96))
    stride = int(cfg.get("stride", 64))
    normalize = cfg.get("normalize", "zscore")

    ds_train = TriPlaneQSM(Vtr, Mtr, Ctr, l=l, patch=patch, stride=stride, normalize=normalize)
    ds_val   = TriPlaneQSM(Vva, Mva, Cva, l=l, patch=patch, stride=stride, normalize=normalize)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    base = int(cfg.get("base", 64))
    csar_rank = int(cfg.get("csar_rank", 4))
    csar_T = float(cfg.get("csar_T", 1.0))
    sem_hidden_ratio = float(cfg.get("sem_hidden_ratio", 0.5))

    model = TriPlaneModel(
        L=l, base=base, share_weights=share_branch,
        csar_rank=csar_rank, csar_T=csar_T, sem_hidden_ratio=sem_hidden_ratio
    ).to(device)

    # Optimizer / Scaler / Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_cfg = cfg.get("loss", {})
    loss_fn = QSMLoss(
        w_l1=float(loss_cfg.get("w_l1", 1.0)),
        w_grad=float(loss_cfg.get("w_grad", 0.05)),
        w_hfen=float(loss_cfg.get("w_hfen", 0.10)),
        w_freq=float(loss_cfg.get("w_freq", 0.05)),
        w_ssim=float(loss_cfg.get("w_ssim", 0.05)),
        ssim_window=int(loss_cfg.get("ssim_window", 11)),
        ssim_sigma=float(loss_cfg.get("ssim_sigma", 1.5)),
        hfen_sigma=float(loss_cfg.get("hfen_sigma", 1.0)),
    )

    # Book-keeping
    best = {"epoch": -1, "nRMSE": float("inf"), "T": csar_T}
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    # Training loop
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, dl_train, optimizer, scaler, loss_fn, device,
            use_amp=use_amp, tri_cfg=tri_cfg, epoch=ep
        )

        # Validation
        val_metrics = validate(model, dl_val, device)

        # TTA-lite: calibrate T on val set
        if tta_enabled and tta_candidates:
            best_T = calibrate_tta_temperature(model, dl_val, device, candidates=tta_candidates)
            # Re-evaluate with best T (optional; calibrate already sets it)
            val_metrics = validate(model, dl_val, device)
        else:
            best_T = model.ax.csar.slice.T  # current T

        # Log
        msg = (f"Epoch {ep:03d} | train_loss {train_stats['loss']:.6f} | "
               f"{log_metrics('val', val_metrics)} | T*={best_T:.3f} | "
               f"time {time.time()-t0:.1f}s")
        print(msg)

        # Save periodic
        if (ep % save_every) == 0:
            save_checkpoint(out_dir, ep, model, optimizer, scaler, best, cfg)

        # Save best by nRMSE
        if val_metrics["nRMSE"] < best["nRMSE"]:
            best.update({"epoch": ep, "nRMSE": val_metrics["nRMSE"], "T": best_T})
            save_checkpoint(out_dir, ep, model, optimizer, scaler, best, cfg, tag="best")

    # Final summary
    print(f"[Done] Best nRMSE {best['nRMSE']:.6f} at epoch {best['epoch']} with T={best['T']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
