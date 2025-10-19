# -*- coding: utf-8 -*-
"""
Tri-plane 2.5D dataset for QSM (stable v0.9)

This dataset:
- Builds sliding-window 3D patches from full volumes
- For each 3D patch, returns three L-slice stacks along axial/coronal/sagittal planes
- Each stack is cropped to the same P×P window consistent with the 3D patch location
- Handles out-of-bound slice indices by clamping (edge replication)
- Returns tensors with shapes:
    ax   : [L, 1, P, P]
    cor  : [L, 1, P, P]
    sag  : [L, 1, P, P]
    mask : [1, P, P, P]
    chi  : [1, P, P, P]
Where DataLoader adds the batch dimension -> [B, ...].

Author: Your Team
License: MIT (or match your project)
"""
from __future__ import annotations
from typing import List, Sequence, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset


def _ensure_float32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.float32:
        return arr.astype(np.float32, copy=False)
    return arr


def _check_shape(vol: np.ndarray, name: str) -> None:
    if vol.ndim != 3:
        raise ValueError(f"{name} must be a 3D array [D,H,W], got shape {vol.shape}.")


def _crop2d(arr2d: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    """Crop a 2D array to size×size with given top-left. Assumes indices are valid."""
    return arr2d[top:top + size, left:left + size]


def _clamp_index(idx: int, lo: int, hi: int) -> int:
    """Clamp index to [lo, hi] inclusive bounds."""
    return max(lo, min(hi, idx))


class TriPlaneQSM(Dataset):
    """
    Tri-plane 2.5D dataset.

    Parameters
    ----------
    vols : Sequence[np.ndarray]
        List of local-field (or input) volumes, each [D,H,W], float or convertible to float32.
    masks : Sequence[np.ndarray]
        List of brain masks aligned to vols, [D,H,W], values in {0,1} or soft [0..1].
    chis : Sequence[np.ndarray]
        List of ground-truth susceptibility maps aligned to vols, [D,H,W].
    l : int
        Number of stacked slices per plane (odd number, e.g., 11 or 15). Center slice corresponds to the patch center.
    patch : int
        Patch size (P). 3D patch is P×P×P, and each 2D slice crop is P×P.
    stride : int
        Sliding window stride for each dimension when enumerating patches.
    normalize : Optional[str]
        Normalization for input `vols`. One of {None, 'zscore', 'minmax'}. Applied per-volume.
    """

    def __init__(
        self,
        vols: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        chis: Sequence[np.ndarray],
        l: int = 11,
        patch: int = 96,
        stride: int = 64,
        normalize: Optional[str] = None,
    ) -> None:
        super().__init__()
        if len(vols) == 0:
            raise ValueError("Empty input list for vols.")
        if not (len(vols) == len(masks) == len(chis)):
            raise ValueError("vols/masks/chis must have the same length.")

        if l % 2 != 1 or l < 1:
            raise ValueError(f"'l' must be odd and >=1, got {l}.")
        if patch <= 0 or stride <= 0:
            raise ValueError(f"patch and stride must be positive, got patch={patch}, stride={stride}.")

        # Validate and store volumes
        self.vols: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.chis: List[np.ndarray] = []

        for i in range(len(vols)):
            v = _ensure_float32(vols[i])
            m = _ensure_float32(masks[i])
            c = _ensure_float32(chis[i])
            _check_shape(v, f"vols[{i}]")
            _check_shape(m, f"masks[{i}]")
            _check_shape(c, f"chis[{i}]")
            if v.shape != m.shape or v.shape != c.shape:
                raise ValueError(f"Shape mismatch at index {i}: vol {v.shape}, mask {m.shape}, chi {c.shape}.")
            self.vols.append(v)
            self.masks.append(m)
            self.chis.append(c)

        self.l = int(l)
        self.half = l // 2
        self.patch = int(patch)
        self.stride = int(stride)
        self.normalize = normalize

        # Precompute patch start indices across all subjects
        self.indices: List[Tuple[int, int, int, int]] = []  # (subj, z, y, x)
        for si, v in enumerate(self.vols):
            D, H, W = v.shape
            # Ensure at least one patch per dim
            z_starts = list(range(0, max(D - self.patch, 0) + 1, self.stride)) or [0]
            y_starts = list(range(0, max(H - self.patch, 0) + 1, self.stride)) or [0]
            x_starts = list(range(0, max(W - self.patch, 0) + 1, self.stride)) or [0]
            for z0 in z_starts:
                for y0 in y_starts:
                    for x0 in x_starts:
                        self.indices.append((si, z0, y0, x0))

        # Precompute simple per-volume normalization stats (if enabled)
        self._stats: List[Tuple[float, float]] = []  # (mean, std) or (min, max)
        for i, v in enumerate(self.vols):
            if normalize is None:
                self._stats.append((0.0, 1.0))
            elif normalize == "zscore":
                mu = float(v.mean())
                sigma = float(v.std() + 1e-6)
                self._stats.append((mu, sigma))
            elif normalize == "minmax":
                vmin = float(v.min())
                vmax = float(v.max())
                if vmax <= vmin + 1e-6:
                    vmax = vmin + 1e-6
                self._stats.append((vmin, vmax))
            else:
                raise ValueError(f"Unknown normalize option: {normalize}")

    def __len__(self) -> int:
        return len(self.indices)

    def _norm(self, v: np.ndarray, si: int) -> np.ndarray:
        if self.normalize is None:
            return v
        a, b = self._stats[si]
        if self.normalize == "zscore":
            return (v - a) / b
        elif self.normalize == "minmax":
            return (v - a) / (b - a)
        return v

    def _stack_slices_for_plane(
        self,
        vol: np.ndarray,
        plane_axis: int,
        center_idx: int,
        top: int,
        left: int,
        size: int,
    ) -> np.ndarray:
        """
        Build an L-slice stack for a given plane.

        plane_axis: 0 (axial: z), 1 (coronal: y), 2 (sagittal: x)
        center_idx: center index along that axis (we stack [center-half, ..., center+half])
        top, left: top-left cropping for the other two axes
        size: crop size (P)
        Returns: [L, size, size] float32
        """
        D, H, W = vol.shape
        out = np.empty((self.l, size, size), dtype=np.float32)
        for i, offs in enumerate(range(center_idx - self.half, center_idx + self.half + 1)):
            if plane_axis == 0:       # axial -> slice is [H,W], indexed by z
                z = _clamp_index(offs, 0, D - 1)
                sl = vol[z, :, :]
                crop = _crop2d(sl, top, left, size)
            elif plane_axis == 1:     # coronal -> slice is [D,W], indexed by y
                y = _clamp_index(offs, 0, H - 1)
                sl = vol[:, y, :]
                crop = _crop2d(sl, top, left, size)
            else:                      # sagittal -> slice is [D,H], indexed by x
                x = _clamp_index(offs, 0, W - 1)
                sl = vol[:, :, x]
                crop = _crop2d(sl, top, left, size)
            out[i] = crop
        return out

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        si, z0, y0, x0 = self.indices[i]

        vol  = self._norm(self.vols[si], si)
        mask = self.masks[si]
        chi  = self.chis[si]

        D, H, W = vol.shape
        P = self.patch

        # Center indices for the 3D patch
        zc = z0 + P // 2
        yc = y0 + P // 2
        xc = x0 + P // 2

        # Crop 3D cubes (inputs for 3D losses/metrics)
        # These are all guaranteed valid because z0,y0,x0 were enumerated to fit patch size
        patch_mask = mask[z0:z0 + P, y0:y0 + P, x0:x0 + P]
        patch_chi  = chi [z0:z0 + P, y0:y0 + P, x0:x0 + P]

        # Build L-slice stacks for three planes, each cropped to the P×P window
        ax_stack  = self._stack_slices_for_plane(vol,  0, zc, y0, x0, P)  # axial slices cropped by (y0,x0)
        cor_stack = self._stack_slices_for_plane(vol,  1, yc, z0, x0, P)  # coronal slices cropped by (z0,x0)
        sag_stack = self._stack_slices_for_plane(vol,  2, xc, z0, y0, P)  # sagittal slices cropped by (z0,y0)

        # Add channel dim=1 to 2D stacks; add channel dim=1 to 3D cubes
        ax  = torch.from_numpy(ax_stack[None, ...])      # [1,L,P,P]
        cor = torch.from_numpy(cor_stack[None, ...])     # [1,L,P,P]
        sag = torch.from_numpy(sag_stack[None, ...])     # [1,L,P,P]

        # Re-order to [L,1,P,P] as agreed with our branch input
        ax  = ax.permute(1, 0, 2, 3).contiguous()        # [L,1,P,P]
        cor = cor.permute(1, 0, 2, 3).contiguous()
        sag = sag.permute(1, 0, 2, 3).contiguous()

        mask3d = torch.from_numpy(patch_mask[None, ...]) # [1,P,P,P]
        chi3d  = torch.from_numpy(patch_chi [None, ...]) # [1,P,P,P]

        return {
            "ax":   ax,          # [L,1,P,P]  -> DataLoader adds batch -> [B,L,1,P,P]
            "cor":  cor,
            "sag":  sag,
            "mask": mask3d,      # [1,P,P,P]
            "chi":  chi3d,       # [1,P,P,P]
            "meta": torch.tensor([si, z0, y0, x0, P, self.l], dtype=torch.int32),
        }
