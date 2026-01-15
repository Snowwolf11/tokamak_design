"""
grids.py
========

R-Z grid utilities for axisymmetric tokamak calculations.

Purpose
-------
Provide a consistent way to build the computational grid used by:
• Grad–Shafranov solvers
• Vacuum field / coil Green's function precomputations
• Postprocessing (contours, field derivs, etc.)

Conventions
-----------
• 1D coordinate arrays:
    R: shape (NR,)
    Z: shape (NZ,)
• 2D mesh arrays (for field storage):
    RR, ZZ: shape (NZ, NR)
  Note the ordering: (Z index first, then R index).
  This is consistent with typical image-like storage and contouring.

Units
-----
All lengths are SI meters [m].
"""

from typing import Tuple
import numpy as np


def make_rz_grid(
    R_min: float,
    R_max: float,
    Z_min: float,
    Z_max: float,
    NR: int,
    NZ: int,
    *,
    endpoint: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a uniform rectangular R-Z grid.

    Parameters
    ----------
    R_min, R_max : float
        Radial bounds [m]. Must satisfy R_max > R_min.
        NOTE: Avoid including R=0 in tokamak calculations.
    Z_min, Z_max : float
        Vertical bounds [m]. Must satisfy Z_max > Z_min.
    NR, NZ : int
        Number of grid points in R and Z. Must be >= 2.
    endpoint : bool
        Passed through to numpy.linspace. True means include endpoints.

    Returns
    -------
    R : np.ndarray, shape (NR,)
        1D radial coordinate array [m]
    Z : np.ndarray, shape (NZ,)
        1D vertical coordinate array [m]
    RR : np.ndarray, shape (NZ, NR)
        2D meshgrid of R values [m]
    ZZ : np.ndarray, shape (NZ, NR)
        2D meshgrid of Z values [m]

    Notes
    -----
    Uses np.meshgrid(indexing="xy") and returns arrays shaped (NZ, NR).
    With indexing="xy":
        RR varies along axis=1 (R direction)
        ZZ varies along axis=0 (Z direction)
    """
    _validate_grid_inputs(R_min, R_max, Z_min, Z_max, NR, NZ)

    R = np.linspace(R_min, R_max, NR, endpoint=endpoint, dtype=float)
    Z = np.linspace(Z_min, Z_max, NZ, endpoint=endpoint, dtype=float)

    # With indexing="xy":
    # RR, ZZ have shape (NZ, NR) given (R, Z) inputs.
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    return R, Z, RR, ZZ


def grid_spacing(R: np.ndarray, Z: np.ndarray) -> Tuple[float, float]:
    """
    Compute uniform grid spacings (dR, dZ).

    Parameters
    ----------
    R : np.ndarray, shape (NR,)
        1D R coordinate array
    Z : np.ndarray, shape (NZ,)
        1D Z coordinate array

    Returns
    -------
    dR : float
        Radial grid spacing [m]
    dZ : float
        Vertical grid spacing [m]

    Raises
    ------
    ValueError if grid is not strictly increasing or not (approximately) uniform.

    Notes
    -----
    In v1 we assume uniform grids.
    If you later add non-uniform grids, you can:
      • return arrays dR[i], dZ[j], or
      • provide a separate function grid_spacing_nonuniform(...)
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if R.ndim != 1 or Z.ndim != 1:
        raise ValueError("R and Z must be 1D arrays.")

    if R.size < 2 or Z.size < 2:
        raise ValueError("R and Z must have at least 2 points.")

    if not np.all(np.diff(R) > 0):
        raise ValueError("R must be strictly increasing.")
    if not np.all(np.diff(Z) > 0):
        raise ValueError("Z must be strictly increasing.")

    dR_arr = np.diff(R)
    dZ_arr = np.diff(Z)

    dR = float(np.mean(dR_arr))
    dZ = float(np.mean(dZ_arr))

    # Check approximate uniformity (tolerances can be tuned)
    if not np.allclose(dR_arr, dR, rtol=1e-10, atol=0.0):
        raise ValueError("R grid is not uniform (diffs not constant).")
    if not np.allclose(dZ_arr, dZ, rtol=1e-10, atol=0.0):
        raise ValueError("Z grid is not uniform (diffs not constant).")

    return dR, dZ


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _validate_grid_inputs(R_min: float, R_max: float, Z_min: float, Z_max: float, NR: int, NZ: int) -> None:
    """Validate input parameters for make_rz_grid()."""
    if not (np.isfinite(R_min) and np.isfinite(R_max) and np.isfinite(Z_min) and np.isfinite(Z_max)):
        raise ValueError("Grid bounds must be finite numbers.")

    if R_max <= R_min:
        raise ValueError(f"Require R_max > R_min, got {R_max} <= {R_min}")
    if Z_max <= Z_min:
        raise ValueError(f"Require Z_max > Z_min, got {Z_max} <= {Z_min}")

    if NR < 2 or NZ < 2:
        raise ValueError("NR and NZ must be >= 2.")

    # Strong hint: avoid R=0
    if R_min <= 0.0:
        # Not strictly forbidden (some tests might want it), but warn by error for tokamaks
        raise ValueError(
            "R_min <= 0 detected. Tokamak GS / axisymmetric formulas typically assume R>0.\n"
            "Choose R_min > 0 to avoid singularities."
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing grids.py")

    R, Z, RR, ZZ = make_rz_grid(0.5, 2.5, -1.0, 1.0, NR=6, NZ=5)

    assert R.shape == (6,)
    assert Z.shape == (5,)
    assert RR.shape == (5, 6)
    assert ZZ.shape == (5, 6)

    dR, dZ = grid_spacing(R, Z)
    print("R:", R)
    print("Z:", Z)
    print("dR, dZ:", dR, dZ)

    print("grids.py self-test passed")
