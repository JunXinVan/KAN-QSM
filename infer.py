# -*- coding: utf-8 -*-
"""
Inference script for Tri-plane 2.5D QSM (stable v1.0)

- Loads a single volume + mask
- Runs sliding-window inference with tri-plane 2.5D branches and 3D fusion
- Overlap-add stitching with mask as weights
- Saves full-volume prediction (.npy). If nibabel is available and output ends with .nii/.nii.gz,
  saves NIfTI as well (assumes identity affine unless --like_nifti is provided).

Usage:
  python infer.py \
    --ckpt checkpoints/epoch_0050_best.pth \
    --branch simple \
    --vol  /path/subject_vol.npy \
    --mask /path/subject_mask.npy \
    --out  /path/pred_subject.npy \
    --l 11 --patch 96 --stride 64 --normalize zscore --T 1.0 --batch 1 --amp

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
import os, argparse
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local modules
from data.datasets import TriPlaneQSM
from models.triplane_fusion3d import Fusion3D, embed_three
from models.branch2d_ukan import Simple2DBranch
from models.ukan_branch import UKAN2DBranch


# ------------------------------
# IO helpers
# ------------------------------
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
        raise ValueError(f"Unsupported file extension for {path}. Use .npy or .npz.")
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array [D,H,W], got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def maybe_save_nifti(volume: np.ndarray, out_path: str, like_path: Optional[str] = None) -> bool:
    """
    Try to save NIfTI if nibabel is installed and out_path has nii/nii.gz suffix.
    If like_path is given and is a NIfTI file, copy its affine/header.
    Returns True if NIfTI was saved, else False.
    """
    if not (out_path.endswith(".nii") or out_path.endswith(".nii.gz")):
        return False
    try:
        import nibabel as nib  # optional
    except Exception:
        return False
    affine = np.eye(4, dtype=np.float64)
    header = None
    if like_path and (like_path.endswith(".nii") or like_path.endswith(".nii.gz")) and os.path.exists(like_path):
        try:
            ref = nib.load(like_path)
            affine = ref.affine
            header = ref.header
        except Exception:
            pass
    img = nib.Nifti1Image(volume.astype(np.float32), affine, header=header)
    nib.save(img, out_path)
    return True


# ------------------------------
# Model (eval-time wrapper)
# ------------------------------
class TriPlaneModelEval(nn.Module):
    """
    Three 2D branches (ax/cor/sag) + 3D fusion head.
    Switchable branch: Simple2DBranch or UKAN2DBranch.
    """
    def __init__(self, L: int, branch: str = "simple",
                 base: int = 64, share_weights: bool = True,
                 csar_rank: int = 4, csar_T: float = 1.0,
                 sem_hidden_ratio: float = 0.5) -> None:
        super().__init__()
        if branch == "ukan":
            mk_branch = lambda: UKAN2DBranch(L=L, T=csar_T, rank=csar_rank)
        else:
            mk_branch = lambda: Simple2DBranch(L=L, in_ch=1, base=base,
                                               csar_rank=csar_rank, csar_T=csar_T,
                                               sem_hidden_ratio=sem_hidden_ratio)
        if share_weights:
            self.ax = self.cor = self.sag = mk_branch()
        else:
            self.ax, self.cor, self.sag = mk_branch(), mk_branch(), mk_branch()
        self.fuse3d = Fusion3D(in_ch=3, base=32)

    @torch.no_grad()
    def set_slice_temperature(self, T: float) -> None:
        if hasattr(self.ax, "csar"):  # Simple2DBranch path
            self.ax.csar.slice.T = float(T); self.cor.csar.slice.T = float(T); self.sag.csar.slice.T = float(T)
        if hasattr(self.ax, "slice_fuser"):  # UKAN2DBranch path
            self.ax.slice_fuser.T = float(T); self.cor.slice_fuser.T = float(T); self.sag.slice_fuser.T = float(T)

    def forward(self, ax: torch.Tensor, cor: torch.Tensor, sag: torch.Tensor,
                mask3d: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_ax  = self.ax(ax)    # [B,1,P,P]
        y_cor = self.cor(cor)
        y_sag = self.sag(sag)
        v_ax, v_cor, v_sag = embed_three(y_ax, y_cor, y_sag, sigma_ratio=0.125)
        pred = self.fuse3d(torch.cat([v_ax, v_cor, v_sag], dim=1), mask=mask3d)
        return pred, dict(vol_ax=v_ax, vol_cor=v_cor, vol_sag=v_sag)


# ------------------------------
# Stitch helpers
# ------------------------------
def stitch_init(shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    D, H, W = shape
    return np.zeros((D, H, W), dtype=np.float32), np.zeros((D, H, W), dtype=np.float32)


def stitch_add(pred_sum: np.ndarray, w_sum: np.ndarray,
               pred_patch: np.ndarray, w_patch: np.ndarray,
               z0: int, y0: int, x0: int) -> None:
    P = pred_patch.shape[0]
    sl = (slice(z0, z0 + P), slice(y0, y0 + P), slice(x0, x0 + P))
    pred_sum[sl] += pred_patch
    w_sum[sl] += w_patch


def stitch_finalize(pred_sum: np.ndarray, w_sum: np.ndarray) -> np.ndarray:
    return pred_sum / (w_sum + 1e-6)


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pth)")
    ap.add_argument("--branch", type=str, choices=["simple", "ukan"], default="simple")
    ap.add_argument("--vol", type=str, required=True, help="Input volume (.npy/.npz)")
    ap.add_argument("--mask", type=str, required=True, help="Brain mask (.npy/.npz)")
    ap.add_argument("--out", type=str, required=True, help="Output path (.npy or .nii/.nii.gz if nibabel installed)")
    ap.add_argument("--like_nifti", type=str, default=None, help="Optional reference NIfTI for affine/header")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--l", type=int, default=11, help="Stacked slices per plane (odd)")
    ap.add_argument("--patch", type=int, default=96, help="Patch size")
    ap.add_argument("--stride", type=int, default=64, help="Stride")
    ap.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])
    ap.add_argument("--base", type=int, default=64, help="Base channels for simple branch")
    ap.add_argument("--csar_rank", type=int, default=4)
    ap.add_argument("--T", type=float, default=1.0, help="Slice-attention temperature")
    ap.add_argument("--share", action="store_true", help="Share branch weights")
    ap.add_argument("--batch", type=int, default=1, help="Patch batch size for inference")
    ap.add_argument("--amp", action="store_true", help="Enable AMP for inference")
    args = ap.parse_args()

    device = torch.device(args.device)
    # Load arrays
    vol  = load_array(args.vol)
    mask = load_array(args.mask)
    if vol.shape != mask.shape:
        raise ValueError(f"vol/mask shape mismatch: {vol.shape} vs {mask.shape}")

    # Build dataset with dummy chi (zeros). We only need patch enumeration + mask.
    chi_dummy = np.zeros_like(vol, dtype=np.float32)
    ds = TriPlaneQSM([vol], [mask], [chi_dummy],
                     l=args.l, patch=args.patch, stride=args.stride,
                     normalize=(None if args.normalize == "none" else args.normalize))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    # Build model
    model = TriPlaneModelEval(
        L=args.l, branch=args.branch, base=args.base, share_weights=args.share,
        csar_rank=args.csar_rank, csar_T=args.T, sem_hidden_ratio=0.5
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:   print(f"[Warn] Missing keys: {missing}")
    if unexpected:print(f"[Warn] Unexpected keys: {unexpected}")

    # Optionally set T (overrides ckpt)
    model.set_slice_temperature(args.T)

    # Stitch accumulators
    pred_sum, w_sum = stitch_init(vol.shape)

    model.eval()
    use_amp = bool(args.amp)
    with torch.no_grad():
        for batch in dl:
            ax  = batch["ax"].to(device).float()    # [B,L,1,P,P]
            cor = batch["cor"].to(device).float()
            sag = batch["sag"].to(device).float()
            m3d = batch["mask"].to(device).float()  # [B,1,P,P,P]
            meta = batch["meta"].cpu().numpy()      # [B,6]

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, _ = model(ax, cor, sag, m3d)  # [B,1,P,P,P]

            pred_np = pred.squeeze(1).cpu().numpy()     # [B,P,P,P]
            w_np    = m3d.squeeze(1).cpu().numpy()      # [B,P,P,P]
            for b in range(pred_np.shape[0]):
                si, z0, y0, x0, P, L = map(int, meta[b])
                # si (subject index) should be 0 here since we passed a single volume
                stitch_add(pred_sum, w_sum, pred_np[b], w_np[b], z0, y0, x0)

    out_vol = stitch_finalize(pred_sum, w_sum)

    # Save .npy
    if args.out.endswith(".npy"):
        np.save(args.out, out_vol.astype(np.float32))
        print(f"[OK] Saved NPY to {args.out}")
    elif args.out.endswith(".npz"):
        np.savez_compressed(args.out, data=out_vol.astype(np.float32))
        print(f"[OK] Saved NPZ to {args.out}")
    else:
        # Try NIfTI if possible
        ok = maybe_save_nifti(out_vol, args.out, like_path=args.like_nifti)
        if ok:
            print(f"[OK] Saved NIfTI to {args.out}")
        else:
            # Fallback to .npy if extension not supported
            np.save(args.out + ".npy", out_vol.astype(np.float32))
            print(f"[OK] Saved NPY to {args.out+'.npy'} (NIfTI not available or bad extension)")

if __name__ == "__main__":
    main()
