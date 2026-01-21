"""
tokdesign.physics.derived
========================

Derived 1D-ish equilibrium quantities from grid-based fields.

This module provides pragmatic Stage-01 implementations for:
- compute_shear(q, rho)
- compute_alpha(rho, q, dp_drho)

These are *not* full flux-surface-averaged, straight-field-line definitions.
They are robust proxies meant to support early optimization and constraint
development.

Terminology / conventions
-------------------------
rho:
    A normalized radial coordinate. In Stage-01 we use:
        rho = sqrt(psi_bar)
    so rho ∈ [0,1] inside the plasma.

q:
    Safety factor. In Stage-01, q is currently computed on the (R,Z) grid
    and then (often) interpreted as q(psi_bar) via binning/averaging.
    Therefore, q may be:
      - 2D on the grid, or
      - already reduced to 1D as a function of rho.

s (magnetic shear proxy):
    Standard definition (in many texts):
        s = (r/q) dq/dr
    With rho as proxy for r/a (dimensionless), a robust approximation is:
        s(rho) = (rho/q) dq/d rho
    This is what we implement.

alpha (s-α model proxy):
    In ballooning theory, a common definition is:
        alpha = - (2 μ0 R q^2 / B^2) dp/dr
    For Stage-01 we don't yet have a consistent 1D B^2(rho) and R(rho).
    We therefore implement a *dimensionless proxy*:
        alpha(rho) = C * q(rho)^2 * (- dp/d rho)
    where C is a tunable constant (default 1.0) that sets the scale.

Why this is acceptable early on
-------------------------------
- Optimization needs stable, monotone, differentiable surrogates.
- Many constraints in early design are *relative* (e.g., avoid negative shear
  in the edge band, keep alpha small where q is low, etc.).
- Once you add proper flux-surface averaging, replace alpha with the
  physically normalized expression (and update metrics thresholds accordingly).

Public API
----------
compute_shear(q, rho) -> s
compute_alpha(rho, q, dp_drho, alpha_scale=1.0, clip=...) -> alpha
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Helpers: reduction to 1D (rho)
# =============================================================================

def _to_1d_profile(
    x: np.ndarray,
    rho: np.ndarray,
    *,
    nbins: int = 80,
    rmin: float = 0.0,
    rmax: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an array `x` (possibly 2D) into a 1D profile x(rho) by bin-averaging.

    If x and rho are already 1D and same shape, we return them as-is (sorted).

    Parameters
    ----------
    x, rho : ndarray
        Either:
          - both 1D (N,), or
          - both same-shape (e.g. 2D grid).
    nbins : int
        Number of rho bins for reduction if x is not 1D.
    rmin, rmax : float
        Bin edges span.

    Returns
    -------
    rho_c : ndarray (nbins,)
        Bin centers (sorted).
    x_c : ndarray (nbins,)
        Mean x in each bin, NaNs filled by interpolation where possible.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Case 1: already 1D profile
    if x.ndim == 1 and rho.ndim == 1 and x.shape == rho.shape:
        # sort by rho
        order = np.argsort(rho)
        return rho[order], x[order]

    # Case 2: reduce via binning
    if x.shape != rho.shape:
        raise ValueError(f"x shape {x.shape} must match rho shape {rho.shape} for reduction.")

    rr = rho.ravel()
    xx = x.ravel()

    edges = np.linspace(rmin, rmax, int(nbins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full((nbins,), np.nan, dtype=float)

    idx = np.digitize(rr, edges) - 1
    ok = (idx >= 0) & (idx < nbins) & np.isfinite(rr) & np.isfinite(xx)

    for k in range(nbins):
        sel = ok & (idx == k)
        if np.any(sel):
            means[k] = float(np.mean(xx[sel]))

    # Fill NaNs by interpolation
    valid = np.isfinite(means)
    if np.sum(valid) >= 2:
        means = np.interp(centers, centers[valid], means[valid])
    elif np.sum(valid) == 1:
        means[:] = means[valid][0]
    else:
        means[:] = np.nan

    return centers, means


# =============================================================================
# Public: shear
# =============================================================================

def compute_shear(
    q: np.ndarray,
    rho: np.ndarray,
    *,
    eps_q: float = 1e-12,
    nbins: int = 80,
) -> np.ndarray:
    """
    Compute magnetic shear proxy s = (rho/q) dq/d rho.

    Accepts q and rho either as:
    - 1D profiles (same shape)
    - 2D grids (same shape), in which case a 1D profile is obtained via binning.

    Returns
    -------
    s : ndarray
        Same shape as input if input is 1D;
        if input is 2D, returns a 2D array mapped back via rho bin interpolation.
    """
    q = np.asarray(q, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Reduce to 1D if needed
    rho_1d, q_1d = _to_1d_profile(q, rho, nbins=nbins)

    # Guard against pathological q (zeros/NaNs)
    q_safe = np.where(np.isfinite(q_1d), np.maximum(q_1d, eps_q), np.nan)

    dq_drho = np.gradient(q_safe, rho_1d + 1e-30, edge_order=1)
    s_1d = (rho_1d / q_safe) * dq_drho

    # If original input was 1D, return 1D
    if q.ndim == 1 and rho.ndim == 1 and q.shape == rho.shape:
        return s_1d.astype(float)

    # Otherwise, map back to grid by interpolating s(rho)
    s_grid = np.full_like(q, np.nan, dtype=float)
    rr = np.clip(rho, rho_1d[0], rho_1d[-1])
    # np.interp expects 1D arrays
    s_grid.flat[:] = np.interp(rr.ravel(), rho_1d, s_1d).astype(float)
    return s_grid


# =============================================================================
# Public: alpha
# =============================================================================

def compute_alpha(
    *,
    rho: np.ndarray,
    q: np.ndarray,
    dp_drho: np.ndarray,
    alpha_scale: float = 1.0,
    clip: Optional[Tuple[float, float]] = None,
    nbins: int = 80,
) -> np.ndarray:
    """
    Compute a Stage-01 proxy for the s-α "alpha" parameter.

    We implement:
        alpha(rho) = alpha_scale * q(rho)^2 * (- dp/d rho)

    Why this form?
    - In many stability models alpha is proportional to q^2 * pressure gradient.
    - We keep it dimensionless-ish and robust for optimization.
    - You can later replace alpha_scale with a physically normalized
      prefactor using <R/B^2> etc from flux surface averages.

    Inputs may be 1D or 2D; if 2D, they are reduced to 1D via binning, then
    mapped back to the original shape.

    Parameters
    ----------
    rho : ndarray
        Normalized radius, typically sqrt(psi_bar), in [0,1].
    q : ndarray
        Safety factor (grid or profile).
    dp_drho : ndarray
        Pressure derivative w.r.t rho. If provided as grid-like values, we
        reduce it to 1D consistently with q.
    alpha_scale : float
        Prefactor.
    clip : (min,max) or None
        Optional clip bounds to prevent extreme values during optimization.
    nbins : int
        Number of bins used if reduction is required.

    Returns
    -------
    alpha : ndarray
        Same shape convention as compute_shear.
    """
    rho = np.asarray(rho, dtype=float)
    q = np.asarray(q, dtype=float)
    dp_drho = np.asarray(dp_drho, dtype=float)

    # Reduce each to 1D consistently on rho bins
    rho_1d, q_1d = _to_1d_profile(q, rho, nbins=nbins)
    _, dp_1d = _to_1d_profile(dp_drho, rho, nbins=nbins)

    # alpha proxy
    alpha_1d = float(alpha_scale) * (q_1d ** 2) * (-dp_1d)

    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        alpha_1d = np.clip(alpha_1d, lo, hi)

    # If original inputs were 1D profiles, return 1D
    if rho.ndim == 1 and q.ndim == 1 and dp_drho.ndim == 1 and rho.shape == q.shape == dp_drho.shape:
        return alpha_1d.astype(float)

    # Map back to grid
    alpha_grid = np.full_like(q, np.nan, dtype=float)
    rr = np.clip(rho, rho_1d[0], rho_1d[-1])
    alpha_grid.flat[:] = np.interp(rr.ravel(), rho_1d, alpha_1d).astype(float)

    return alpha_grid
