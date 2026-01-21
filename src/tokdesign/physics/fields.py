"""
tokdesign.physics.fields
=======================

Magnetic field reconstruction from the Grad–Shafranov flux function ψ(R,Z)
and the toroidal flux function F(ρ) ≡ R B_φ.

This module is intentionally small, robust, and Stage-01 friendly.

Conventions
-----------
Axisymmetric magnetic field is represented as:

    B = B_R e_R + B_Z e_Z + B_φ e_φ

Given poloidal flux ψ(R,Z) (a scalar field on an (R,Z) grid), the poloidal field
components are:

    B_R = - (1/R) ∂ψ/∂Z
    B_Z = + (1/R) ∂ψ/∂R

Given the toroidal flux function F, where:

    F(R,Z) = R * B_φ(R,Z)

we reconstruct:

    B_φ = F / R

Important notes
--------------
• Inputs:
    - R, Z are 1D coordinate arrays
    - psi is a 2D array with shape (NZ, NR), matching meshgrid indexing:
        RR, ZZ = np.meshgrid(R, Z)  -> psi[iz, ir] corresponds to (R[ir], Z[iz])
• We use centered finite differences in the interior and 1st-order at the edges.
• We include safe handling near R=0 (although tokamak grids should never include R=0).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def poloidal_field_from_psi(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    *,
    eps_R: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (B_R, B_Z) from ψ on a rectilinear (R,Z) grid.

    Parameters
    ----------
    R, Z : ndarray
        1D coordinate arrays. R has length NR, Z has length NZ.
    psi : ndarray
        2D array of ψ with shape (NZ, NR).
    eps_R : float
        Small floor for R in the 1/R factors to avoid division by zero.

    Returns
    -------
    BR, BZ : ndarray
        2D arrays with shape (NZ, NR).
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    psi = np.asarray(psi, dtype=float)

    if psi.ndim != 2:
        raise ValueError(f"psi must be 2D (NZ, NR), got shape {psi.shape}")
    NZ, NR = psi.shape
    if R.ndim != 1 or Z.ndim != 1:
        raise ValueError("R and Z must be 1D arrays.")
    if len(R) != NR:
        raise ValueError(f"len(R)={len(R)} does not match psi.shape[1]={NR}")
    if len(Z) != NZ:
        raise ValueError(f"len(Z)={len(Z)} does not match psi.shape[0]={NZ}")

    # Grid spacings (assume uniform; Stage-01 uses uniform)
    # If non-uniform grids appear later, replace with np.gradient(..., R, axis=1) etc.
    dR = float(R[1] - R[0]) if NR > 1 else 1.0
    dZ = float(Z[1] - Z[0]) if NZ > 1 else 1.0
    if dR == 0 or dZ == 0:
        raise ValueError("Degenerate grid spacing: dR or dZ is zero.")

    # ∂ψ/∂R (axis=1), ∂ψ/∂Z (axis=0)
    dpsi_dR = _ddx_2d(psi, dR, axis=1)
    dpsi_dZ = _ddx_2d(psi, dZ, axis=0)

    # Build RR grid (broadcast-friendly)
    RR = R[None, :]  # shape (1, NR), broadcasts over Z
    RR_safe = np.maximum(RR, eps_R)

    BR = -(1.0 / RR_safe) * dpsi_dZ
    BZ = +(1.0 / RR_safe) * dpsi_dR

    return BR, BZ


def toroidal_field_from_F(
    RR: np.ndarray,
    F: np.ndarray,
    *,
    eps_R: float = 1e-12,
) -> np.ndarray:
    """
    Compute toroidal field Bphi from F = R * Bphi.

    Parameters
    ----------
    RR : ndarray
        2D array of R coordinates (same shape as F).
    F : ndarray
        2D array of F values. Often computed as F(rho(R,Z)).
    eps_R : float
        Small floor for R to avoid division by zero.

    Returns
    -------
    Bphi : ndarray
        2D array of Bphi, same shape as F.
    """
    RR = np.asarray(RR, dtype=float)
    F = np.asarray(F, dtype=float)

    if RR.shape != F.shape:
        raise ValueError(f"RR shape {RR.shape} must match F shape {F.shape}")

    RR_safe = np.maximum(RR, eps_R)
    return F / RR_safe


def magnetic_field_magnitude(
    BR: np.ndarray,
    BZ: np.ndarray,
    Bphi: np.ndarray,
) -> np.ndarray:
    """
    Convenience helper: |B| = sqrt(BR^2 + BZ^2 + Bphi^2).
    """
    BR = np.asarray(BR, dtype=float)
    BZ = np.asarray(BZ, dtype=float)
    Bphi = np.asarray(Bphi, dtype=float)
    if not (BR.shape == BZ.shape == Bphi.shape):
        raise ValueError("BR, BZ, Bphi must all have the same shape.")
    return np.sqrt(BR * BR + BZ * BZ + Bphi * Bphi)


# =============================================================================
# Finite difference utilities
# =============================================================================

def _ddx_2d(a: np.ndarray, dx: float, *, axis: int) -> np.ndarray:
    """
    First derivative of a 2D array along `axis` using:
    - centered differences in the interior
    - 1st-order one-sided differences at boundaries

    Parameters
    ----------
    a : ndarray
        2D array.
    dx : float
        Grid spacing.
    axis : int
        0 for Z-like axis (rows), 1 for R-like axis (cols).

    Returns
    -------
    da_dx : ndarray
        Same shape as `a`.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError("_ddx_2d expects a 2D array.")
    if dx == 0:
        raise ValueError("dx must be non-zero.")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1.")

    da = np.empty_like(a, dtype=float)

    if axis == 0:
        # derivative along rows (Z direction)
        da[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dx)
        da[0, :] = (a[1, :] - a[0, :]) / dx
        da[-1, :] = (a[-1, :] - a[-2, :]) / dx
    else:
        # derivative along cols (R direction)
        da[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
        da[:, 0] = (a[:, 1] - a[:, 0]) / dx
        da[:, -1] = (a[:, -1] - a[:, -2]) / dx

    return da
