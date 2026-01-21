"""
tokdesign.physics.boundary
==========================

LCFS boundary generation utilities.

Stage-01 uses a *prescribed* plasma boundary (fixed-boundary equilibrium).
This module turns shaping controls (R0, a, kappa, delta, Z0, ...) into a
closed polyline (N,2) with columns [R, Z].

Implemented models
------------------
1) "miller-ish" parametric boundary (default)
   A simple, robust shaping parameterization using:
     - major radius R0
     - minor radius a
     - elongation kappa
     - triangularity delta (standard Miller-style shift)
     - vertical shift Z0
     - optional "squareness" / higher harmonics (future)

This is NOT intended to be a perfect Miller equilibrium fit; it is a convenient
and differentiable boundary generator for early pipeline stages.

Public API
----------
build_lcfs_polyline_from_controls(controls, n=256) -> polyline (N,2), closed
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def build_lcfs_polyline_from_controls(
    controls: Dict[str, Any],
    *,
    n: int = 256,
) -> np.ndarray:
    """
    Construct a closed LCFS polyline from `controls`.

    Expected Stage-01 controls layout (typical)
    ------------------------------------------
    controls["plasma_boundary"] may contain:
        R0     : float   major radius [m]
        a      : float   minor radius [m]
        Z0     : float   vertical shift [m] (default 0)
        kappa  : float   elongation (>=1 typically)
        delta  : float   triangularity (usually 0..0.6)
        model  : str     "miller" (default) or "ellipse"

        n      : int     optional override for point count

    If controls["plasma_boundary"]["polyline"] exists, we return it (after ensuring
    it is closed). That allows loading boundaries from file or upstream code.

    Parameters
    ----------
    controls : dict
        Stage-01 controls.
    n : int
        Number of points for the polyline (including closure point if added).
        Must be >= 32 for reasonable shape.

    Returns
    -------
    poly : ndarray, shape (N,2)
        Closed polyline with columns [R, Z] and poly[0]==poly[-1].
    """
    if not isinstance(controls, dict):
        raise TypeError("controls must be a dict.")

    pb = controls.get("plasma_boundary", {})
    if not isinstance(pb, dict):
        pb = {}

    # If user already provides a polyline, use it.
    if "polyline" in pb and pb["polyline"] is not None:
        poly = np.asarray(pb["polyline"], dtype=float)
        return _ensure_closed_polyline(poly)

    # Point count
    n_pb = int(pb.get("n", n))
    if n_pb < 32:
        raise ValueError("LCFS polyline needs at least 32 points for stability/quality.")

    model = str(pb.get("model", "miller")).lower().strip()

    # Common parameters with safe defaults
    R0 = float(pb.get("R0", 1.7))
    a = float(pb.get("a", 0.5))
    Z0 = float(pb.get("Z0", 0.0))
    kappa = float(pb.get("kappa", 1.7))
    delta = float(pb.get("delta", 0.0))

    if a <= 0:
        raise ValueError("plasma_boundary.a must be > 0.")
    if R0 <= 0:
        raise ValueError("plasma_boundary.R0 must be > 0.")
    if kappa <= 0:
        raise ValueError("plasma_boundary.kappa must be > 0.")

    if model in ("ellipse", "elliptic"):
        poly = _ellipse_boundary(R0=R0, a=a, kappa=kappa, Z0=Z0, n=n_pb)
    elif model in ("miller", "millerish", "miller-ish"):
        poly = _millerish_boundary(R0=R0, a=a, kappa=kappa, delta=delta, Z0=Z0, n=n_pb)
    else:
        raise ValueError(f"Unknown plasma_boundary.model={model!r}. Use 'miller' or 'ellipse'.")

    return _ensure_closed_polyline(poly)


# =============================================================================
# Shape models
# =============================================================================

def _ellipse_boundary(*, R0: float, a: float, kappa: float, Z0: float, n: int) -> np.ndarray:
    """
    Simple ellipse boundary:
        R = R0 + a cosθ
        Z = Z0 + κ a sinθ
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    R = R0 + a * np.cos(theta)
    Z = Z0 + (kappa * a) * np.sin(theta)
    return np.column_stack([R, Z])


def _millerish_boundary(
    *,
    R0: float,
    a: float,
    kappa: float,
    delta: float,
    Z0: float,
    n: int,
) -> np.ndarray:
    """
    A robust "Miller-ish" parametric boundary.

    A common Miller parameterization is:
        R(θ) = R0 + a cos(θ + sinθ * δ)
        Z(θ) = Z0 + κ a sinθ

    This produces:
    - δ > 0 shifts the upper/lower regions inward (triangularity)
    - κ stretches vertically

    Notes
    -----
    • This is not the full Miller model (which also includes squareness via harmonics).
    • This is stable for optimization and stage-01 usage.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    # Clamp to a sane range to avoid pathological shapes early in optimization.
    # You can loosen/remove these later if you want.
    delta_c = float(np.clip(delta, -0.95, 0.95))
    kappa_c = float(np.clip(kappa, 0.2, 10.0))

    R = R0 + a * np.cos(theta + delta_c * np.sin(theta))
    Z = Z0 + (kappa_c * a) * np.sin(theta)

    return np.column_stack([R, Z])


# =============================================================================
# Utilities
# =============================================================================

def _ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    """
    Ensure polyline is shape (N,2) and closed (first point repeated at end).
    """
    poly = np.asarray(poly, dtype=float)

    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError(f"polyline must have shape (N,2), got {poly.shape}")
    if poly.shape[0] < 4:
        raise ValueError("polyline must have at least 4 points.")

    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    return poly
