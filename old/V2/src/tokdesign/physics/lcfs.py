"""
physics/lcfs.py
===============

Utilities for extracting and handling the Last Closed Flux Surface (LCFS)
from a poloidal flux field ψ(R, Z). (compared to plasma_boundary.py which creates target shape)

This module is geometry- and field-based only:
- no GS solving
- no coils
- no optimization
- no I/O

It is designed for robust use inside free-boundary equilibrium iteration.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path


# =============================================================================
# Exceptions
# =============================================================================

class LCFSExtractionError(RuntimeError):
    """Raised when a valid LCFS cannot be extracted."""
    pass

# =============================================================================
# Public API
# =============================================================================

def extract_lcfs(
    RR: np.ndarray,
    ZZ: np.ndarray,
    psi: np.ndarray,
    psi_lcfs: float,
    axis_RZ: tuple[float, float] | None = None,
    selector: str = "enclosing_axis",
    n_resample: int | None = None,
) -> np.ndarray:
    """
    Extract the LCFS contour ψ = ψ_lcfs from a 2D ψ field.

    Parameters
    ----------
    RR, ZZ : 2D arrays
        Meshgrid arrays of R and Z coordinates.
    psi : 2D array
        Total poloidal flux ψ(R, Z).
    psi_lcfs : float
        Flux value defining the LCFS.
    axis_RZ : (R, Z), optional
        Magnetic axis location. Strongly recommended for robust selection.
    selector : str
        Contour selection strategy. Currently supported:
          - "enclosing_axis" (default)
          - "largest_area"
    n_resample : int, optional
        If given, resample the LCFS to this many points with uniform arc length.

    Returns
    -------
    lcfs : (N, 2) ndarray
        LCFS polyline with columns [R, Z], ordered counter-clockwise.

    Raises
    ------
    LCFSExtractionError
        If no suitable LCFS contour can be found.
    """

    # -------------------------------------------------------------------------
    # 1. Extract all contours at ψ = ψ_lcfs
    # -------------------------------------------------------------------------

    contours = _find_contours(RR, ZZ, psi, psi_lcfs)

    if not contours:
        raise LCFSExtractionError(
            "No ψ = ψ_lcfs contours found in domain."
        )

    # -------------------------------------------------------------------------
    # 2. Select the physically correct contour
    # -------------------------------------------------------------------------

    if selector == "enclosing_axis":
        if axis_RZ is None:
            raise LCFSExtractionError(
                "axis_RZ must be provided for selector='enclosing_axis'."
            )
        lcfs = _select_contour_enclosing_point(contours, axis_RZ)

    elif selector == "largest_area":
        lcfs = max(contours, key=_polygon_area)

    else:
        raise ValueError(f"Unknown LCFS selector '{selector}'")

    if lcfs is None:
        raise LCFSExtractionError(
            "Failed to select a valid LCFS contour."
        )

    # -------------------------------------------------------------------------
    # 3. Enforce orientation and basic sanity
    # -------------------------------------------------------------------------

    lcfs = _enforce_ccw(lcfs)

    if lcfs.shape[0] < 10:
        raise LCFSExtractionError(
            "Extracted LCFS has too few points."
        )

    # -------------------------------------------------------------------------
    # 4. Optional resampling
    # -------------------------------------------------------------------------

    if n_resample is not None:
        lcfs = resample_lcfs(lcfs, n_resample)

    return lcfs


# =============================================================================
# Contour extraction helpers
# =============================================================================

def _find_contours(
    RR: np.ndarray,
    ZZ: np.ndarray,
    psi: np.ndarray,
    level: float,
) -> list[np.ndarray]:
    """
    Find all closed ψ-contours at a given level.

    Returns a list of (N,2) arrays in physical (R,Z) coordinates.
    """

    # Use matplotlib's contouring (robust and well-tested)
    fig = plt.figure()
    try:
        cs = plt.contour(RR, ZZ, psi, levels=[level])
    finally:
        plt.close(fig)

    contours = []

    for collection in cs.collections:
        for path in collection.get_paths():
            verts = path.vertices  # (N,2) array [R,Z]

            # Require closed contour
            if verts.shape[0] < 4:
                continue

            if not np.allclose(verts[0], verts[-1]):
                # matplotlib contours are usually closed, but be explicit
                verts = np.vstack([verts, verts[0]])

            contours.append(verts)

    return contours


# =============================================================================
# Contour selection helpers
# =============================================================================

def _select_contour_enclosing_point(
    contours: list[np.ndarray],
    point: tuple[float, float],
) -> np.ndarray | None:
    """
    Select the contour that encloses a given point (R,Z).
    """

    R0, Z0 = point

    for poly in contours:
        path = Path(poly)
        if path.contains_point((R0, Z0)):
            return poly

    return None


# =============================================================================
# Geometry utilities
# =============================================================================

def _polygon_area(poly: np.ndarray) -> float:
    """
    Signed polygon area using the shoelace formula.
    """
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def _enforce_ccw(poly: np.ndarray) -> np.ndarray:
    """
    Ensure polygon is counter-clockwise.
    """
    if _polygon_area(poly) < 0.0:
        return poly[::-1]
    return poly


def resample_lcfs(
    poly: np.ndarray,
    npts: int,
) -> np.ndarray:
    """
    Resample a closed LCFS polyline to npts points with uniform arc length.
    """

    # Remove duplicate last point if present
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]

    # Arc length parameterization
    d = np.sqrt(np.sum(np.diff(poly, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    s /= s[-1]

    # Interpolate
    s_new = np.linspace(0.0, 1.0, npts, endpoint=False)

    R_new = np.interp(s_new, s, poly[:, 0])
    Z_new = np.interp(s_new, s, poly[:, 1])

    lcfs_new = np.column_stack([R_new, Z_new])

    return lcfs_new


# =============================================================================
# Self-test
# =============================================================================

def _selftest_extract_lcfs():
    """
    Self-test for LCFS extraction.

    Uses an analytic circular flux function:
        ψ = (R - R0)^2 + Z^2

    LCFS is a circle of radius a.
    """

    print("Running LCFS self-test...")

    # -------------------------------------------------------------------------
    # Construct grid
    # -------------------------------------------------------------------------

    R0 = 1.7
    a = 0.5

    R = np.linspace(1.0, 2.4, 200)
    Z = np.linspace(-0.9, 0.9, 240)
    RR, ZZ = np.meshgrid(R, Z)

    # -------------------------------------------------------------------------
    # Analytic ψ field
    # -------------------------------------------------------------------------

    psi = (RR - R0)**2 + ZZ**2
    psi_lcfs = a**2

    axis = (R0, 0.0)

    # Add small noise to mimic numerical GS solution
    rng = np.random.default_rng(1)
    psi += 1e-5 * rng.standard_normal(psi.shape)

    # -------------------------------------------------------------------------
    # Extract LCFS
    # -------------------------------------------------------------------------

    lcfs = extract_lcfs(
        RR,
        ZZ,
        psi,
        psi_lcfs,
        axis_RZ=axis,
        n_resample=128,
    )

    # -------------------------------------------------------------------------
    # Basic sanity checks
    # -------------------------------------------------------------------------

    assert lcfs.shape == (128, 2), "Unexpected LCFS shape"

    # Distance from analytic circle
    r = np.sqrt((lcfs[:, 0] - R0)**2 + lcfs[:, 1]**2)
    r_mean = np.mean(r)
    r_err = np.max(np.abs(r - a))

    print(f"  Mean radius     : {r_mean:.6f}")
    print(f"  Max radius error: {r_err:.3e}")

    assert abs(r_mean - a) < 5e-3, "Mean LCFS radius incorrect"
    assert r_err < 1e-2, "LCFS deviation too large"

    # Orientation check (CCW)
    area = _polygon_area(np.vstack([lcfs, lcfs[0]]))
    assert area > 0.0, "LCFS orientation is not counter-clockwise"

    print("LCFS self-test PASSED.")


# Allow direct execution
if __name__ == "__main__":
    _selftest_extract_lcfs()
