# -*- coding: utf-8 -*-
"""
Train script for Tri-plane 2.5D QSM using UKAN2DBranch (stable v1.0)

Pipeline:
  - Dataset: TriPlaneQSM -> [ax, cor, sag] slice stacks + [mask, chi]
  - Model  : three UKAN2DBranch (shared or separate) + 3D fusion head
  - Loss   : QSMLoss (+ optional Tri-consistency among three embedded volumes)
  - Val    : nRMSE / HFEN(LoG) / SSIM(3D) / Freq-L1(rFFT)
  - TTA-lite: scan slice-attention temperature T on val set from candidate list
  - Save   : periodic checkpoints + best-by-nRMSE checkpoint

Usage:
  python train_ukan.py --config configs/ukan.yaml

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
import os, argparse, time, random
from typing import Dict, Any, Tuple, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except Exception:
    tqdm = None
import yaml

# Local modules
from data.datasets import TriPlaneQSM
from models.ukan_branch import UKAN2DBranch
from models.triplane_fusion3d import (
    Fusion3D, embed_three, tri_consistency_loss
)
from losses.qsm_losses import QSMLoss, hfen_log3d, freq_l1_rfft3d, ssim3d


# ------------------------------
# Utils
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
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
    elif path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            key = "data" if "data" in z.files else z.files[0]
            arr = z[key]
    else:
        raise ValueError(f"Unsupported file extension for {path}, use .npy or .npz")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W] from {path}, got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def load_split_lists(cfg: Dict[str, Any]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray],
                                                   List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    if "data" not in cfg or "train" not in cfg["data"] or "val" not in cfg["data"]:
        raise ValueError("Config must define data.train and data.val with vols/masks/chis lists.")
    def _read(split: Dict[str, Any]):
        vols = split.get("vols", [])
        masks = split.get("masks", [])
        chis = split.get("chis", [])
        if not (len(vols) == len(masks) == len(chis) and len(vols) > 0):
            raise ValueError("Each split must provide non-empty equal-length lists: vols/masks/chis.")
        V = [load_array(p) for p in vols]
        M = [load_array(p) for p in masks]
        C = [load_array(p) for p in chis]
        return V, M, C
    return _read(cfg["data"]["train"]) + _read(cfg["data"]["val"])


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True).float()
        else:
            out[k] = v
    return out


def masked_nrmse(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.unsqueeze(1)
        pred = pred * mask
        gt = gt * mask
    num = torch.sqrt(torch.clamp((pred - gt).pow(2).sum(), min=1e-12))
    den = torch.sqrt(torch.clamp((gt).pow(2).sum(), min=1e-12))
    return (num / den).item()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, float]:
    with torch.no_grad():
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


def log_metrics(tag: str, m: Dict[str, float]) -> str:
    return f"{tag} | nRMSE {m['nRMSE']:.6f} | HFEN {m['HFEN']:.6f} | SSIM {m['SSIM']:.6f} | FreqL1 {m['FreqL1']:.6f}"


# ------------------------------
# Model (UKAN branch)
# ------------------------------
class TriPlaneModelUKAN(nn.Module):
    """
    Three UKAN2DBranch (ax/cor/sag) + 3D fusion.
    """
    def __init__(self, L: int = 11, share_weights: bool = True,
                 csar_rank: int = 4, csar_T: float = 1.0,
                 ukan_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        def Branch():
            return UKAN2DBranch(L=L, T=csar_T, rank=csar_rank,
                                ukan_kwargs=ukan_kwargs, force_rgb=True)
        if share_weights:
            self.ax = self.cor = self.sag = Branch()
        else:
            self.ax, self.cor, self.sag = Branch(), Branch(), Branch()
        self.fuse3d = Fusion3D(in_ch=3, base=32)

    @torch.no_grad()
    def set_slice_temperature(self, T: float) -> None:
        self.ax.set_slice_temperature(T)
        self.cor.set_slice_temperature(T)
        self.sag.set_slice_temperature(T)

    def forward(self, ax: torch.Tensor, cor: torch.Tensor, sag: torch.Tensor,
                mask3d: torch.Tensor):
        y_ax  = self.ax(ax)   # [B,1,P,P]
        y_cor = self.cor(cor)
        y_sag = self.sag(sag)
        vol_ax, vol_cor, vol_sag = embed_three(y_ax, y_cor, y_sag, sigma_ratio=0.125)
        pred3d = self.fuse3d(torch.cat([vol_ax, vol_cor, vol_sag], dim=1), mask=mask3d)
        aux = dict(y_ax=y_ax, y_cor=y_cor, y_sag=y_sag,
                   vol_ax=vol_ax, vol_cor=vol_cor, vol_sag=vol_sag)
        return pred3d, aux


# ------------------------------
# Train / Val / TTA-lite
# ------------------------------
def train_one_epoch(
    model: TriPlaneModelUKAN,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    loss_fn: QSMLoss,
    device: torch.device,
    use_amp: bool,
    tri_cfg: Dict[str, Any],
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cnt = 0

    tri_weight = 0.0
    tri_mode = tri_cfg.get("mode", "l1")
    if tri_cfg.get("enabled", True):
        start_ep = tri_cfg.get("start_epoch", total_epochs // 2)
        if epoch >= start_ep:
            tri_weight = float(tri_cfg.get("weight", 0.02))

    iterator = tqdm(loader, desc=f"Train[{epoch}]") if tqdm else loader
    for batch in iterator:
        batch = to_device(batch, device)
        ax, cor, sag = batch["ax"], batch["cor"], batch["sag"]   # [B,L,1,P,P]
        mask, gt     = batch["mask"], batch["chi"]               # [B,1,P,P,P]

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred, aux = model(ax, cor, sag, mask)
            loss = loss_fn(pred, gt, mask)
            if tri_weight > 0.0:
                loss = loss + tri_weight * tri_consistency_loss(
                    aux["vol_ax"], aux["vol_cor"], aux["vol_sag"], mask=mask, mode=tri_mode
                )

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
        total_cnt  += bs

    return {"loss": total_loss / max(1, total_cnt)}


@torch.no_grad()
def validate(model: TriPlaneModelUKAN, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = dict(nRMSE=0.0, HFEN=0.0, SSIM=0.0, FreqL1=0.0)
    cnt = 0
    iterator = tqdm(loader, desc="Val") if tqdm else loader
    for batch in iterator:
        batch = to_device(batch, device)
        ax, cor, sag = batch["ax"], batch["cor"], batch["sag"]
        mask, gt     = batch["mask"], batch["chi"]
        pred, _ = model(ax, cor, sag, mask)
        m = compute_metrics(pred, gt, mask)
        for k in sums: sums[k] += m[k] * ax.size(0)
        cnt += ax.size(0)
    cnt = max(1, cnt)
    return {k: v / cnt for k, v in sums.items()}


@torch.no_grad()
def calibrate_tta_temperature(
    model: TriPlaneModelUKAN,
    loader: DataLoader,
    device: torch.device,
    candidates: Sequence[float],
) -> float:
    best_T, best_score = float(candidates[0]), float("inf")
    for T in candidates:
        model.set_slice_temperature(T)
        m = validate(model, loader, device)
        if m["nRMSE"] < best_score:
            best_score = m["nRMSE"]
            best_T = float(T)
    model.set_slice_temperature(best_T)
    return best_T


def save_checkpoint(
    out_dir: str, epoch: int,
    model: TriPlaneModelUKAN, optimizer: optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best: Dict[str, Any], cfg: Dict[str, Any], tag: str = ""
) -> None:
    ensure_dir(out_dir)
    state = dict(
        epoch=epoch, model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=(scaler.state_dict() if scaler is not None else None),
        best=best, cfg=cfg
    )
    fname = f"epoch_{epoch:04d}{('_'+tag) if tag else ''}.pth"
    torch.save(state, os.path.join(out_dir, fname))


# ------------------------------
# Main
# ------------------------------
def main(cfg_path: str) -> None:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Seeds & device
    seed = int(cfg.get("seed", 2025)); set_seed(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = cfg.get("out_dir", "checkpoints_ukan"); ensure_dir(out_dir)

    # Data
    Vtr, Mtr, Ctr, Vva, Mva, Cva = load_split_lists(cfg)
    l      = int(cfg.get("l", 11))
    patch  = int(cfg.get("patch", 96))
    stride = int(cfg.get("stride", 64))
    normalize = cfg.get("normalize", "zscore")

    ds_train = TriPlaneQSM(Vtr, Mtr, Ctr, l=l, patch=patch, stride=stride, normalize=normalize)
    ds_val   = TriPlaneQSM(Vva, Mva, Cva, l=l, patch=patch, stride=stride, normalize=normalize)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 1))
    epochs     = int(train_cfg.get("epochs", 50))
    lr         = float(train_cfg.get("lr", 2.0e-4))
    wd         = float(train_cfg.get("weight_decay", 1.0e-4))
    use_amp    = bool(train_cfg.get("amp", True))
    share_branch = bool(train_cfg.get("share_branch_weights", True))
    save_every = int(train_cfg.get("save_every", 5))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # UKAN kwargs (optional; safe defaults in UKAN2DBranch if empty)
    ukan_kwargs = cfg.get("ukan", None)

    # CSAR settings (slice attention inside UKAN2DBranch)
    csar_rank = int(cfg.get("csar_rank", 4))
    csar_T    = float(cfg.get("csar_T", 1.0))

    # Build model
    model = TriPlaneModelUKAN(
        L=l, share_weights=share_branch,
        csar_rank=csar_rank, csar_T=csar_T,
        ukan_kwargs=ukan_kwargs
    ).to(device)

    # Optimizer / scaler / loss
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

    # Tri-consistency
    tri_cfg = cfg.get("tri_consistency", {})
    if "start_epoch" not in tri_cfg:
        tri_cfg["start_epoch"] = epochs // 2

    # TTA-lite
    tta_cfg = cfg.get("tta_lite", {"enabled": True, "candidates": [0.8, 1.0, 1.2]})
    tta_enabled    = bool(tta_cfg.get("enabled", True))
    tta_candidates = tta_cfg.get("candidates", [0.8, 1.0, 1.2])

    # Save config copy
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    best = {"epoch": -1, "nRMSE": float("inf"), "T": csar_T}

    # Loop
    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, dl_train, optimizer, scaler, loss_fn, device,
            use_amp=use_amp, tri_cfg=tri_cfg, epoch=ep, total_epochs=epochs
        )
        val_metrics = validate(model, dl_val, device)

        # TTA-lite
        if tta_enabled and tta_candidates:
            bestT = calibrate_tta_temperature(model, dl_val, device, candidates=tta_candidates)
            val_metrics = validate(model, dl_val, device)
        else:
            bestT = csar_T  # current

        # Log
        msg = (f"Epoch {ep:03d} | train_loss {train_stats['loss']:.6f} | "
               f"{log_metrics('val', val_metrics)} | T*={bestT:.3f} | "
               f"time {time.time()-t0:.1f}s")
        print(msg)

        # Save periodic
        if (ep % save_every) == 0:
            save_checkpoint(out_dir, ep, model, optimizer, scaler, best, cfg)

        # Save best
        if val_metrics["nRMSE"] < best["nRMSE"]:
            best.update({"epoch": ep, "nRMSE": val_metrics["nRMSE"], "T": bestT})
            save_checkpoint(out_dir, ep, model, optimizer, scaler, best, cfg, tag="best")

    print(f"[Done] Best nRMSE {best['nRMSE']:.6f} at epoch {best['epoch']} with T={best['T']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ukan.yaml")
    args = parser.parse_args()
    main(args.config)
