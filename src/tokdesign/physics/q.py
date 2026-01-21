"""
tokdesign.physics.q
===================

Safety-factor (q) profile estimation for axisymmetric equilibria.

This module provides a pragmatic, Stage-01-friendly implementation of
`compute_q_profile(...)` used by `equilibrium.py`.

Physics background (what we compute)
------------------------------------
For an axisymmetric equilibrium, one common definition is:

    q(ψ) = (1 / 2π) ∮ (B · ∇φ) / (B · ∇θ) dθ
         = (1 / 2π) ∮ (Bφ / (R Bp)) dl_p

Using:
    Bφ = F / R
    Bp = |∇ψ| / R

we get the convenient contour integral form:
    q(ψ) = (1 / 2π) ∮ F / (R |∇ψ|) dl

where the line integral is taken along a poloidal flux surface ψ = const.

Implementation approach
-----------------------
1) Compute ψ_bar = normalized flux in [0,1]
2) Extract several closed contours ψ_bar = level using matplotlib's contouring
3) For each contour:
     - sample F, R, and |∇ψ| along the curve by bilinear interpolation
     - compute q(level) via the contour integral
4) Interpolate q(level) back onto the grid as q(ψ_bar(R,Z))

This is not the fastest approach, but it is:
• robust for early-stage development
• easy to validate visually
• requires no straight-field-line coordinates

If contour extraction fails (e.g. extremely coarse grid), we fall back to a
simple heuristic approximation (documented below).

Dependencies
------------
numpy
scipy.interpolate.RegularGridInterpolator
matplotlib (for contour extraction)

Notes
-----
- The function returns q on the full (Z,R) grid (same shape as ψ).
- Outside the LCFS, q is set to NaN by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path as MplPath
import matplotlib
matplotlib.use("Agg")  # safe for headless execution
import matplotlib.pyplot as plt


# =============================================================================
# Public API
# =============================================================================

def compute_q_profile(
    *,
    RR: np.ndarray,
    ZZ: np.ndarray,
    psi: np.ndarray,
    F: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    lcfs_poly: np.ndarray,
    n_levels: int = 25,
    level_min: float = 0.05,
    level_max: float = 0.95,
    min_points_per_contour: int = 40,
) -> np.ndarray:
    """
    Compute q on the (Z,R) grid by flux-surface contour integration.

    Parameters
    ----------
    RR, ZZ : ndarray
        2D meshgrids (NZ,NR) as returned by np.meshgrid(R, Z).
    psi : ndarray
        Poloidal flux ψ(R,Z) on the grid (NZ,NR).
    F : ndarray
        Toroidal flux function values F(R,Z) on the grid (NZ,NR).
        (Often constructed as F(rho(ψ_bar)).)
    psi_axis : float
        ψ at magnetic axis.
    psi_lcfs : float
        ψ at LCFS (Dirichlet boundary).
    lcfs_poly : ndarray
        LCFS polyline (N,2) with columns [R,Z], closed or open (we handle).
    n_levels : int
        Number of ψ_bar contour levels to evaluate between level_min and level_max.
    level_min, level_max : float
        Contour levels in ψ_bar space (0..1). Avoid 0 and 1 to reduce numerical issues.
    min_points_per_contour : int
        Reject contours with too few vertices.

    Returns
    -------
    q_grid : ndarray
        q(R,Z) on the grid (NZ,NR). Outside LCFS: NaN.
    """
    RR = np.asarray(RR, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)
    psi = np.asarray(psi, dtype=float)
    F = np.asarray(F, dtype=float)

    if RR.shape != psi.shape or ZZ.shape != psi.shape or F.shape != psi.shape:
        raise ValueError("RR, ZZ, psi, and F must all have the same shape (NZ,NR).")

    NZ, NR = psi.shape
    if NZ < 8 or NR < 8:
        # Too small for meaningful contours; fallback.
        return _fallback_q(RR, ZZ, psi, F, psi_axis, psi_lcfs, lcfs_poly)

    # Grid coordinates
    R = RR[0, :]
    Z = ZZ[:, 0]

    # Mask inside LCFS
    lcfs_poly = _ensure_closed_polyline(np.asarray(lcfs_poly, dtype=float))
    inside = _mask_inside_polyline(RR, ZZ, lcfs_poly)

    # Normalize flux
    psi_bar = _normalize_psi_bar(psi, psi_axis, psi_lcfs)

    # Compute |∇ψ| on grid (finite differences)
    dpsi_dZ, dpsi_dR = np.gradient(psi, Z, R, edge_order=1)  # axis0=Z, axis1=R
    grad_psi_mag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) + 1e-30

    # Interpolators for R, F, |∇ψ|
    # Note: RegularGridInterpolator expects (Z, R) axes in that order.
    F_itp = RegularGridInterpolator((Z, R), F, bounds_error=False, fill_value=np.nan)
    g_itp = RegularGridInterpolator((Z, R), grad_psi_mag, bounds_error=False, fill_value=np.nan)
    R_itp = RegularGridInterpolator((Z, R), RR, bounds_error=False, fill_value=np.nan)

    # Extract contours of psi_bar
    levels = np.linspace(level_min, level_max, int(n_levels))
    q_levels = []
    psi_levels = []

    # Use a dedicated (hidden) figure for contour extraction
    fig = plt.figure(figsize=(4, 4))
    try:
        cs = plt.contour(RR, ZZ, psi_bar, levels=levels)
    except Exception:
        plt.close(fig)
        return _fallback_q(RR, ZZ, psi, F, psi_axis, psi_lcfs, lcfs_poly)
    finally:
        # We still need cs collections below; figure must remain alive until done.
        pass

    # Choose one "best" closed contour per level (prefer the one enclosing the axis)
    axis_RZ = _estimate_axis_point(RR, ZZ, psi, inside, psi_axis)

    for lev_idx, lev in enumerate(levels):
        paths = cs.collections[lev_idx].get_paths()
        if not paths:
            continue

        best = _select_best_closed_path(paths, axis_RZ=axis_RZ, min_points=min_points_per_contour)
        if best is None:
            continue

        pts = best  # (M,2) columns [R,Z]
        # Convert to (Z,R) points for interpolators
        pts_ZR = np.column_stack([pts[:, 1], pts[:, 0]])

        # Sample along contour
        F_s = F_itp(pts_ZR)
        g_s = g_itp(pts_ZR)
        R_s = R_itp(pts_ZR)

        # If too many NaNs, skip
        if np.any(~np.isfinite(F_s)) or np.any(~np.isfinite(g_s)) or np.any(~np.isfinite(R_s)):
            continue

        # Arc-length differential along contour
        dR = np.diff(pts[:, 0], append=pts[0, 0])
        dZ = np.diff(pts[:, 1], append=pts[0, 1])
        dl = np.sqrt(dR**2 + dZ**2)

        # q = (1/2π) ∮ F / (R |∇ψ|) dl
        integrand = F_s / (R_s * g_s)
        q_val = float(np.sum(integrand * dl) / (2.0 * np.pi))

        if np.isfinite(q_val) and q_val > 0.0:
            q_levels.append(q_val)
            psi_levels.append(float(lev))

    plt.close(fig)

    if len(q_levels) < 3:
        # Not enough data for a meaningful profile; fallback.
        return _fallback_q(RR, ZZ, psi, F, psi_axis, psi_lcfs, lcfs_poly)

    psi_levels = np.asarray(psi_levels, dtype=float)
    q_levels = np.asarray(q_levels, dtype=float)

    # Sort by psi level
    order = np.argsort(psi_levels)
    psi_levels = psi_levels[order]
    q_levels = q_levels[order]

    # Build q on grid by interpolation in psi_bar
    q_grid = np.full_like(psi, np.nan, dtype=float)
    pb = psi_bar[inside]
    q_grid[inside] = np.interp(pb, psi_levels, q_levels, left=q_levels[0], right=q_levels[-1])
    import sys
    #np.set_printoptions(threshold=sys.maxsize)
    #print((~np.isnan(q_grid)).sum(axis=1))
    return q_grid


# =============================================================================
# Helpers
# =============================================================================

def _normalize_psi_bar(psi: np.ndarray, psi_axis: float, psi_lcfs: float) -> np.ndarray:
    denom = float(psi_lcfs - psi_axis)
    if abs(denom) < 1e-30:
        return np.zeros_like(psi, dtype=float)
    out = (psi - float(psi_axis)) / denom
    return np.clip(out, 0.0, 1.0)


def _ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("lcfs_poly must have shape (N,2) with columns [R,Z].")
    if poly.shape[0] < 4:
        raise ValueError("lcfs_poly must have at least 4 points.")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _mask_inside_polyline(RR: np.ndarray, ZZ: np.ndarray, poly: np.ndarray) -> np.ndarray:
    path = MplPath(poly)
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    inside = path.contains_points(pts)
    return inside.reshape(RR.shape)


def _estimate_axis_point(
    RR: np.ndarray,
    ZZ: np.ndarray,
    psi: np.ndarray,
    inside: np.ndarray,
    psi_axis: float,
) -> Tuple[float, float]:
    """
    Return (R_axis, Z_axis) by picking the grid point inside LCFS whose ψ is
    closest to psi_axis (usually an extremum inside).
    """
    # If psi_axis is from an argmax, the closest point is likely that argmax.
    masked = np.where(inside, np.abs(psi - psi_axis), np.inf)
    flat = int(np.argmin(masked))
    iz, ir = np.unravel_index(flat, psi.shape)
    return float(RR[iz, ir]), float(ZZ[iz, ir])


def _select_best_closed_path(
    paths,
    *,
    axis_RZ: Tuple[float, float],
    min_points: int,
) -> Optional[np.ndarray]:
    """
    Choose a good closed contour among many candidates.

    Heuristic:
    - keep only closed paths with enough points
    - prefer the one that contains the magnetic axis point
    - among those, prefer the one with the most vertices (typically the main surface)
    """
    axR, axZ = axis_RZ
    best = None
    best_score = -np.inf

    for p in paths:
        v = p.vertices  # (M,2) [R,Z]
        if v.shape[0] < min_points:
            continue
        # Closed if endpoints almost coincide
        if not np.allclose(v[0], v[-1], atol=1e-10, rtol=0.0):
            # matplotlib may return non-closed; we can close it if it's nearly closed
            # or skip. Here: try to close if endpoints are close-ish.
            if np.linalg.norm(v[0] - v[-1]) < 1e-6:
                v = np.vstack([v, v[0]])
            else:
                continue

        # Does it contain the axis?
        try:
            contains_axis = MplPath(v).contains_point((axR, axZ))
        except Exception:
            contains_axis = False

        # Score: axis-containing paths get a big bonus; then by number of points
        score = float(v.shape[0]) + (1e6 if contains_axis else 0.0)

        if score > best_score:
            best_score = score
            best = v

    return best


# =============================================================================
# Fallback (when contour method fails)
# =============================================================================

def _fallback_q(
    RR: np.ndarray,
    ZZ: np.ndarray,
    psi: np.ndarray,
    F: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    lcfs_poly: np.ndarray,
) -> np.ndarray:
    """
    Conservative fallback approximation.

    If we cannot reliably extract contours, we return a "heuristic q" that:
    - is finite inside LCFS
    - increases with radius (via psi_bar)
    - scales with toroidal field magnitude roughly

    This is NOT physically rigorous. It exists so downstream code can run.
    Replace with the contour method once your grid/LCFS extraction is stable.

    q_fallback := q0 + q1 * psi_bar
    where q0 ~ median(F/R)/median(Bp) * geometric_factor

    We approximate Bp via |∇ψ|/R.
    """
    RR = np.asarray(RR, float)
    ZZ = np.asarray(ZZ, float)
    psi = np.asarray(psi, float)
    F = np.asarray(F, float)

    lcfs_poly = _ensure_closed_polyline(np.asarray(lcfs_poly, float))
    inside = _mask_inside_polyline(RR, ZZ, lcfs_poly)

    q = np.full_like(psi, np.nan, dtype=float)
    if not np.any(inside):
        return q

    R = RR[0, :]
    Z = ZZ[:, 0]
    dpsi_dZ, dpsi_dR = np.gradient(psi, Z, R, edge_order=1)
    Bp = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) / np.maximum(RR, 1e-12)

    psi_bar = _normalize_psi_bar(psi, psi_axis, psi_lcfs)

    # robust typical scales inside
    Bp_med = float(np.nanmedian(Bp[inside])) + 1e-30
    Bphi_med = float(np.nanmedian((F / np.maximum(RR, 1e-12))[inside]))

    # crude scale: Bphi/Bp * (a/R0) ~ O(1..10). Use 0.3 as a conservative geometric factor.
    scale = 0.3 * abs(Bphi_med) / Bp_med

    q0 = max(0.2, scale)
    q1 = max(0.2, 1.5 * scale)

    q[inside] = q0 + q1 * psi_bar[inside]
    return q
