"""
fields.py
=========

Magnetic field utilities for axisymmetric tokamak calculations.

Given poloidal flux psi(R,Z) on a uniform R–Z grid, compute the poloidal field:

    BR = -(1/R) * ∂psi/∂Z
    BZ =  (1/R) * ∂psi/∂R

Also provide helpers for toroidal field from F(psi):

    Bphi = F(psi) / R

Conventions (consistent with geometry/grids.py)
-----------------------------------------------
- R: shape (NR,)
- Z: shape (NZ,)
- 2D fields: shape (NZ, NR) with indexing [iz, ir]
- RR, ZZ from np.meshgrid(R, Z, indexing="xy") -> shape (NZ, NR)

Numerics
--------
- Assumes *uniform* grid spacing in R and Z.
- Uses 2nd-order central differences on interior nodes.
- Uses 2nd-order one-sided differences on domain edges.

Notes
-----
- This module is pure numerics/postprocessing. It does not know about masks.
- Use eps_R to guard against accidental R≈0 (though your grid builder forbids R_min<=0).
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional

import numpy as np


def _uniform_spacing(x: np.ndarray, name: str) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 points.")
    if not np.all(np.diff(x) > 0):
        raise ValueError(f"{name} must be strictly increasing.")
    dx_arr = np.diff(x)
    dx = float(np.mean(dx_arr))
    if not np.allclose(dx_arr, dx, rtol=1e-10, atol=0.0):
        raise ValueError(f"{name} grid is not uniform.")
    return dx


def compute_BR_BZ_from_psi(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    *,
    eps_R: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (BR, BZ) from psi on a uniform grid.

    Parameters
    ----------
    R : np.ndarray, shape (NR,)
        Radial grid [m], strictly increasing and uniform.
    Z : np.ndarray, shape (NZ,)
        Vertical grid [m], strictly increasing and uniform.
    psi : np.ndarray, shape (NZ, NR)
        Poloidal flux [Wb/rad] (convention used throughout workflow).
    eps_R : float
        Floor for R in 1/R to avoid division by zero.

    Returns
    -------
    BR : np.ndarray, shape (NZ, NR)
    BZ : np.ndarray, shape (NZ, NR)
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    psi = np.asarray(psi, dtype=float)

    NR = R.size
    NZ = Z.size
    if psi.shape != (NZ, NR):
        raise ValueError(f"psi must have shape (NZ,NR)=({NZ},{NR}), got {psi.shape}")

    dR = _uniform_spacing(R, "R")
    dZ = _uniform_spacing(Z, "Z")

    # Derivatives dpsi/dR and dpsi/dZ with 2nd order accuracy
    dpsi_dR = np.empty_like(psi, dtype=float)
    dpsi_dZ = np.empty_like(psi, dtype=float)

    # ---- d/dR ----
    # interior: central
    dpsi_dR[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dR)
    # left edge: 2nd-order forward
    dpsi_dR[:, 0] = (-3.0 * psi[:, 0] + 4.0 * psi[:, 1] - 1.0 * psi[:, 2]) / (2.0 * dR)
    # right edge: 2nd-order backward
    dpsi_dR[:, -1] = (3.0 * psi[:, -1] - 4.0 * psi[:, -2] + 1.0 * psi[:, -3]) / (2.0 * dR)

    # ---- d/dZ ----
    # interior: central
    dpsi_dZ[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dZ)
    # bottom edge: 2nd-order forward
    dpsi_dZ[0, :] = (-3.0 * psi[0, :] + 4.0 * psi[1, :] - 1.0 * psi[2, :]) / (2.0 * dZ)
    # top edge: 2nd-order backward
    dpsi_dZ[-1, :] = (3.0 * psi[-1, :] - 4.0 * psi[-2, :] + 1.0 * psi[-3, :]) / (2.0 * dZ)

    # Build 2D R mesh efficiently
    RR = np.broadcast_to(R[np.newaxis, :], psi.shape)
    R_safe = np.maximum(RR, float(eps_R))

    BR = -(1.0 / R_safe) * dpsi_dZ
    BZ = (1.0 / R_safe) * dpsi_dR

    return BR, BZ


def compute_Bphi_from_F(
    RR: np.ndarray,
    F: np.ndarray,
    *,
    eps_R: float = 1e-12,
) -> np.ndarray:
    """
    Compute Bphi from F(psi) using Bphi = F / R.

    Parameters
    ----------
    RR : np.ndarray, shape (NZ, NR)
        Radial mesh [m]
    F : np.ndarray, shape (NZ, NR)
        Toroidal field function [T*m] (often equals R*Bphi)
    eps_R : float
        Floor for R in division.

    Returns
    -------
    Bphi : np.ndarray, shape (NZ, NR)
    """
    RR = np.asarray(RR, dtype=float)
    F = np.asarray(F, dtype=float)
    if RR.shape != F.shape:
        raise ValueError(f"RR and F must have the same shape, got {RR.shape} vs {F.shape}")

    R_safe = np.maximum(RR, float(eps_R))
    return F / R_safe


def compute_total_field(
    BR: np.ndarray,
    BZ: np.ndarray,
    Bphi: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience helper to package field components and |B|.

    Parameters
    ----------
    BR, BZ : np.ndarray
        Poloidal components.
    Bphi : Optional[np.ndarray]
        Toroidal component. If None, |B| is computed from BR,BZ only.

    Returns
    -------
    dict with keys:
      - "BR", "BZ", optionally "Bphi"
      - "Bp"  (sqrt(BR^2 + BZ^2))
      - "Bmag" (sqrt(BR^2 + BZ^2 + Bphi^2) or equals Bp if Bphi is None)
    """
    BR = np.asarray(BR, dtype=float)
    BZ = np.asarray(BZ, dtype=float)
    if BR.shape != BZ.shape:
        raise ValueError("BR and BZ must have the same shape.")

    Bp = np.sqrt(BR * BR + BZ * BZ)
    out: Dict[str, np.ndarray] = {"BR": BR, "BZ": BZ, "Bp": Bp}

    if Bphi is None:
        out["Bmag"] = Bp
    else:
        Bphi = np.asarray(Bphi, dtype=float)
        if Bphi.shape != BR.shape:
            raise ValueError("Bphi must have the same shape as BR/BZ.")
        out["Bphi"] = Bphi
        out["Bmag"] = np.sqrt(Bp * Bp + Bphi * Bphi)

    return out


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing fields.py")

    # Uniform grid
    R = np.linspace(1.0, 2.0, 51)
    Z = np.linspace(-0.5, 0.5, 61)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    # Test 1: psi = R^2 => dpsi/dR = 2R => BZ = (1/R)*2R = 2 everywhere, BR = 0
    psi1 = RR ** 2
    BR1, BZ1 = compute_BR_BZ_from_psi(R, Z, psi1)
    assert np.max(np.abs(BR1)) < 1e-12
    assert np.max(np.abs(BZ1 - 2.0)) < 1e-10

    # Test 2: psi = Z^2 => dpsi/dZ = 2Z => BR = -(1/R)*2Z, BZ = 0
    psi2 = ZZ ** 2
    BR2, BZ2 = compute_BR_BZ_from_psi(R, Z, psi2)
    # Ignore very edges (one-sided diffs still fine, but be conservative)
    mid = (slice(2, -2), slice(2, -2))
    BR2_ref = -(2.0 * ZZ[mid]) / RR[mid]
    assert np.max(np.abs(BZ2[mid])) < 1e-10
    assert np.max(np.abs(BR2[mid] - BR2_ref)) < 1e-10

    # Test 3: Bphi from constant F
    F = np.full_like(RR, 4.0)
    Bphi = compute_Bphi_from_F(RR, F)
    assert np.max(np.abs(Bphi[mid] - (4.0 / RR[mid]))) < 1e-14

    print("fields.py self-test passed.")
