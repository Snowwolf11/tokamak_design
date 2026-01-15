"""
plasma_boundary.py
==================

Plasma boundary (LCFS) geometry utilities.

Purpose
-------
Provide simple, reusable tools to:
• generate a Miller-parameterized boundary (common tokamak shaping model)
• compute geometric area of a closed boundary
• estimate shaping parameters (elongation κ and triangularity δ) from a boundary

Conventions
-----------
• A boundary is represented as a polyline array of shape (N, 2):
    boundary[:, 0] = R  [m]
    boundary[:, 1] = Z  [m]

• Boundaries returned by generators are CLOSED:
    boundary[0] == boundary[-1]

Notes
-----
These are purely geometric utilities, not equilibrium solvers.
They are used for:
• defining target LCFS shapes
• deriving shape metrics for /derived and /analysis
• sanity checks and plotting

Miller parameterization (common definition)
-------------------------------------------
Given major radius R0, minor radius a, elongation kappa, triangularity delta:

    R(θ) = R0 + a * cos( θ + delta * sin(θ) )
    Z(θ) = kappa * a * sin(θ)

This yields:
• δ > 0 shifts the top/bottom inward (D-shape)
• κ controls vertical stretching
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


# ============================================================
# GENERATORS
# ============================================================

def miller_boundary(
    R0: float,
    a: float,
    kappa: float,
    delta: float,
    npts: int = 400,
    *,
    closed: bool = True,
) -> np.ndarray:
    """
    Generate a Miller-parameterized boundary (LCFS) polyline.

    Parameters
    ----------
    R0 : float
        Major radius of the plasma centroid [m]
    a : float
        Minor radius (horizontal half-width) [m]
    kappa : float
        Elongation κ = (Z_max - Z_min) / (2a)
    delta : float
        Triangularity δ (dimensionless), typically in [0, ~0.6]
    npts : int
        Number of points along boundary (not counting closure point).
    closed : bool
        If True, append the first point to the end.

    Returns
    -------
    boundary : np.ndarray, shape (npts+1, 2) if closed else (npts, 2)
        Boundary polyline in (R, Z).

    Raises
    ------
    ValueError if parameters are non-physical.
    """
    if npts < 32:
        raise ValueError("npts should be >= 32 for a reasonable boundary.")
    if a <= 0:
        raise ValueError("a must be > 0.")
    if kappa <= 0:
        raise ValueError("kappa must be > 0.")
    if R0 <= 0:
        raise ValueError("R0 must be > 0 (tokamak R-Z convention).")

    theta = np.linspace(0.0, 2.0 * np.pi, npts, endpoint=False)
    R = R0 + a * np.cos(theta + delta * np.sin(theta))
    Z = kappa * a * np.sin(theta)

    boundary = np.column_stack([R, Z])

    if closed:
        boundary = ensure_closed_polyline(boundary)

    return boundary


# ============================================================
# METRICS
# ============================================================

def boundary_area(poly: np.ndarray) -> float:
    """
    Compute the area enclosed by a closed polygon using the shoelace formula.

    Parameters
    ----------
    poly : np.ndarray, shape (N,2)
        Closed boundary polyline.

    Returns
    -------
    area : float
        Enclosed area in the R-Z plane [m^2]

    Notes
    -----
    The area is always returned as a positive value.
    """
    poly = ensure_closed_polyline(poly)
    _validate_polyline(poly)

    x = poly[:, 0]
    y = poly[:, 1]

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return float(area)


def boundary_kappa_delta(poly: np.ndarray) -> Tuple[float, float]:
    """
    Estimate elongation κ and triangularity δ from a boundary polyline.

    Definitions used (standard "engineering" definitions)
    -----------------------------------------------------
    Let:
        R_in  = min R on boundary
        R_out = max R on boundary
        a     = (R_out - R_in)/2
        R0    = (R_out + R_in)/2   (geometric midplane center)

        Z_max = max Z on boundary
        Z_min = min Z on boundary

    Elongation:
        κ = (Z_max - Z_min) / (2a)

    Triangularity:
        Let R_top be R value at Z_max (approx, from the polyline point(s) at max Z)
        Let R_bot be R value at Z_min
        δ_top = (R0 - R_top)/a
        δ_bot = (R0 - R_bot)/a
        δ = (δ_top + δ_bot)/2

    Parameters
    ----------
    poly : np.ndarray, shape (N,2)
        Closed boundary polyline.

    Returns
    -------
    kappa : float
        Estimated elongation
    delta : float
        Estimated triangularity

    Notes
    -----
    • This is a geometric estimate from discrete points. If you want smoother
      estimates, later you can fit a Miller model to the boundary.
    • For X-point / diverted shapes, these simple definitions can be ambiguous.
      For v1, this is fine for "single-valued" D-shapes.
    """
    poly = ensure_closed_polyline(poly)
    _validate_polyline(poly)

    R = poly[:-1, 0]  # ignore duplicate closure point
    Z = poly[:-1, 1]

    R_in = float(np.min(R))
    R_out = float(np.max(R))
    a = 0.5 * (R_out - R_in)
    if a <= 0:
        raise ValueError("Boundary has zero/negative horizontal extent; cannot compute a.")

    R0 = 0.5 * (R_out + R_in)

    Z_max = float(np.max(Z))
    Z_min = float(np.min(Z))

    kappa = (Z_max - Z_min) / (2.0 * a)

    # Estimate R at Z extrema:
    # use average R of points within tolerance of the extremum
    # (helps if multiple points share the max due to discretization)
    tol = 1e-8 * max(1.0, abs(Z_max - Z_min))
    top_mask = np.abs(Z - Z_max) <= tol
    bot_mask = np.abs(Z - Z_min) <= tol

    if np.any(top_mask):
        R_top = float(np.mean(R[top_mask]))
    else:
        # fallback: take the R at the max-Z vertex
        R_top = float(R[np.argmax(Z)])

    if np.any(bot_mask):
        R_bot = float(np.mean(R[bot_mask]))
    else:
        R_bot = float(R[np.argmin(Z)])

    delta_top = (R0 - R_top) / a
    delta_bot = (R0 - R_bot) / a
    delta = 0.5 * (delta_top + delta_bot)

    return float(kappa), float(delta)


# ============================================================
# HELPERS
# ============================================================

def ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    """
    Ensure the polyline is closed by appending the first point at the end if needed.
    """
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("Polyline must have shape (N,2).")
    if poly.shape[0] < 3:
        raise ValueError("Polyline must contain at least 3 points.")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _validate_polyline(poly: np.ndarray) -> None:
    """Basic sanity checks for boundary polylines."""
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("Boundary must have shape (N,2).")
    if poly.shape[0] < 4:
        raise ValueError("Closed boundary should have at least 4 points (including closure).")
    if not np.all(np.isfinite(poly)):
        raise ValueError("Boundary contains non-finite values.")
    if np.any(poly[:, 0] <= 0.0):
        raise ValueError("Boundary contains R <= 0 values (invalid for tokamak R-Z).")


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing plasma_boundary.py")

    R0, a, kappa0, delta0 = 1.65, 0.5, 1.7, 0.35
    poly = miller_boundary(R0, a, kappa0, delta0, npts=800)

    A = boundary_area(poly)
    kappa_est, delta_est = boundary_kappa_delta(poly)

    print("Area [m^2]:", A)
    print("kappa true/est:", kappa0, kappa_est)
    print("delta true/est:", delta0, delta_est)

    # Rough sanity: estimates should be close for a Miller-generated boundary
    assert abs(kappa_est - kappa0) < 1e-2
    assert abs(delta_est - delta0) < 1e-2

    print("plasma_boundary.py self-test passed")
