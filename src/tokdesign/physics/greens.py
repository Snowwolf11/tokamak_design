"""
greens.py
=========

Green's functions for an axisymmetric (toroidal) filamentary circular current loop.

This module provides the core "vacuum field" building block used throughout
the workflow:
• coil psi-per-ampere maps (for inverse coil fitting)
• BR, BZ fields from coil currents
• later: vacuum contributions in free-boundary equilibrium

Physics model (v1)
------------------
Each PF coil is approximated as a thin filamentary circular loop (a current ring)
located at (Rc, Zc) in cylindrical coordinates (R, Z). The configuration is
axisymmetric (no toroidal angle dependence).

The magnetic field of a circular current loop can be expressed using complete
elliptic integrals of the first and second kind.

Conventions
-----------
• Inputs (R, Z) may be scalars or NumPy arrays (broadcasting supported)
• Units:
    R, Z, Rc, Zc in meters [m]
    I in amperes [A]
    BR, BZ in tesla [T]
    psi in webers per radian [Wb/rad] (common GS convention)

Important note on psi
---------------------
In axisymmetric Grad–Shafranov convention, the poloidal flux function psi
is related to the toroidal vector potential A_phi by:
    psi(R,Z) = R * A_phi(R,Z)

This psi is the object used in GS solvers and contouring. It is NOT the total
flux through a physical loop unless you include 2π factors depending on convention.
For this repository we stick to the common GS convention: psi = R A_phi.

Numerical notes
---------------
• We avoid R=0 in the grid (tokamak R>0), because formulas involve division by R.
• Close to the filament (R~Rc, Z~Zc) the field is singular in the filament model.
  That is physical for an ideal filament; real coils have finite cross-section.
  For v1, avoid evaluating exactly on the coil centerline.

Dependencies
------------
We use scipy.special for elliptic integrals if available.
If SciPy is not available, you can add a fallback (mpmath), but for Anaconda
SciPy is typically installed.
"""

from __future__ import annotations

import numpy as np

from tokdesign.constants import MU0


# ============================================================
# ELLIPTIC INTEGRALS BACKEND
# ============================================================

def _ellipk(m):
    """Complete elliptic integral of the first kind K(m)."""
    try:
        from scipy.special import ellipk
        return ellipk(m)
    except Exception as e:
        raise ImportError(
            "SciPy is required for elliptic integral based loop Green's functions.\n"
            "Install with: conda install scipy  (recommended)  or pip install scipy"
        ) from e


def _ellipe(m):
    """Complete elliptic integral of the second kind E(m)."""
    try:
        from scipy.special import ellipe
        return ellipe(m)
    except Exception as e:
        raise ImportError(
            "SciPy is required for elliptic integral based loop Green's functions.\n"
            "Install with: conda install scipy  (recommended)  or pip install scipy"
        ) from e


# ============================================================
# CORE GEOMETRY HELPERS
# ============================================================

def _loop_k2(R: np.ndarray, Z: np.ndarray, Rc: float, Zc: float):
    """
    Compute elliptic parameter k^2 and common geometric quantities.

    Standard definitions:
        k^2 = 4 R Rc / ((R+Rc)^2 + (Z-Zc)^2)
        d   = sqrt((R+Rc)^2 + (Z-Zc)^2)
        rho2 = (R-Rc)^2 + (Z-Zc)^2   (often appears in denominators)

    Returns
    -------
    k2 : ndarray
    d  : ndarray
    rho2 : ndarray
    dz : ndarray
    """
    dz = Z - Zc
    d2 = (R + Rc) ** 2 + dz ** 2
    d = np.sqrt(d2)
    rho2 = (R - Rc) ** 2 + dz ** 2

    # k^2 in [0,1] for physical points (R>0, Rc>0)
    k2 = 4.0 * R * Rc / d2

    # Guard against tiny numerical overshoots (e.g. 1.0000000002)
    k2 = np.clip(k2, 0.0, 1.0)

    return k2, d, rho2, dz


# ============================================================
# PUBLIC API
# ============================================================

def brbz_from_loop(
    R: np.ndarray,
    Z: np.ndarray,
    Rc: float,
    Zc: float,
    I: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute BR and BZ from a filamentary circular loop at (Rc, Zc).

    Parameters
    ----------
    R, Z : array_like
        Evaluation points (broadcastable). Typically RR, ZZ mesh grids.
    Rc, Zc : float
        Coil center coordinates [m]
    I : float
        Current [A]

    Returns
    -------
    BR, BZ : np.ndarray
        Magnetic field components [T], same shape as broadcast(R, Z)

    References (standard formulas)
    ------------------------------
    These expressions can be found in many EM textbooks and notes for the
    magnetic field of a circular loop in cylindrical coordinates, expressed
    with complete elliptic integrals K(k^2), E(k^2).

    Notes
    -----
    • Singular on the filament (ideal model).
    • Requires R>0.
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if np.any(R <= 0.0):
        raise ValueError("brbz_from_loop requires R > 0 everywhere (avoid R=0).")

    k2, d, rho2, dz = _loop_k2(R, Z, Rc, Zc)

    K = _ellipk(k2)
    E = _ellipe(k2)

    # Common prefactor
    # Many published formulas use k (not k^2) and various rearrangements.
    # The following form is widely used and numerically stable for tokamak grids.
    pref = MU0 * I / (2.0 * np.pi)

    # Avoid division by zero in degenerate cases (should not occur if R>0 and not on filament)
    denom = d * rho2 + 1e-300

    # BR expression
    # BR = pref * dz / (R * d) * [ -K + (R^2 + Rc^2 + dz^2)/rho2 * E ]
    BR = pref * dz / (R * d + 1e-300) * (-K + ((R**2 + Rc**2 + dz**2) / (rho2 + 1e-300)) * E)

    # BZ expression
    # BZ = pref / d * [ K + (Rc^2 - R^2 - dz^2)/rho2 * E ]
    BZ = pref / (d + 1e-300) * (K + ((Rc**2 - R**2 - dz**2) / (rho2 + 1e-300)) * E)

    return BR, BZ


def psi_from_loop(
    R: np.ndarray,
    Z: np.ndarray,
    Rc: float,
    Zc: float,
    I: float = 1.0,
) -> np.ndarray:
    """
    Compute poloidal flux function psi(R,Z) = R * A_phi for a filamentary loop.

    Parameters
    ----------
    R, Z : array_like
        Evaluation points (broadcastable).
    Rc, Zc : float
        Coil center coordinates [m]
    I : float
        Current [A]

    Returns
    -------
    psi : np.ndarray
        Poloidal flux function in GS convention [Wb/rad],
        same shape as broadcast(R, Z).

    Formula used
    ------------
    Using the standard expression for the toroidal vector potential A_phi
    of a circular current loop:

        A_phi = (μ0 I / (2π)) * (1/k) * sqrt(Rc/R) * [ (2 - k^2) K(k^2) - 2 E(k^2) ]

    where:
        k^2 = 4 R Rc / ((R+Rc)^2 + (Z-Zc)^2)

    Then:
        psi = R * A_phi

    Notes
    -----
    • Requires R>0.
    • Singular on the filament (ideal model).
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if np.any(R <= 0.0):
        raise ValueError("psi_from_loop requires R > 0 everywhere (avoid R=0).")

    k2, d, _, _ = _loop_k2(R, Z, Rc, Zc)
    K = _ellipk(k2)
    E = _ellipe(k2)

    # k = sqrt(k^2)
    k = np.sqrt(k2)

    pref = MU0 * I / (2.0 * np.pi)

    # Vector potential A_phi for a current loop (axisymmetric)
    # Guard k=0 (far away / on axis limit) by adding epsilon.
    # For k->0, the bracket tends to ~0, so this remains well-behaved.
    eps = 1e-300
    Aphi = pref * (1.0 / (k + eps)) * np.sqrt(Rc / (R + eps)) * ((2.0 - k2) * K - 2.0 * E)

    psi = R * Aphi
    return psi


# ============================================================
# PROJECT-FACING WRAPPERS
# ============================================================

def psi_from_filament_loop(
    RR: np.ndarray,
    ZZ: np.ndarray,
    Rc: float,
    Zc: float,
    I: float = 1.0,
    *,
    method: str = "analytic_elliptic",
) -> np.ndarray:
    """
    Wrapper expected by geometry/coils.py.

    Parameters
    ----------
    RR, ZZ : np.ndarray (NZ, NR)
        Mesh grids
    Rc, Zc : float
        Coil center
    I : float
        Current
    method : str
        Currently only "analytic_elliptic" implemented.

    Returns
    -------
    psi : np.ndarray (NZ, NR)
    """
    method = method.lower()
    if method not in ("analytic_elliptic", "elliptic", "analytic"):
        raise ValueError(f"Unknown greens method: {method}")
    return psi_from_loop(RR, ZZ, Rc=Rc, Zc=Zc, I=I)


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing greens.py")

    # Simple grid around a loop
    R = np.linspace(0.5, 3.0, 200)
    Z = np.linspace(-2.0, 2.0, 201)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    Rc, Zc = 1.5, 0.0

    psi = psi_from_loop(RR, ZZ, Rc, Zc, I=1e5)
    BR, BZ = brbz_from_loop(RR, ZZ, Rc, Zc, I=1e5)

    # Basic sanity checks: shapes and finite away from filament
    assert psi.shape == RR.shape
    assert BR.shape == RR.shape
    assert BZ.shape == RR.shape

    # Check that values are finite at points not on the filament
    # (The filament singularity is at R=Rc, Z=Zc)
    mask_far = (np.abs(RR - Rc) > 1e-3) | (np.abs(ZZ - Zc) > 1e-3)
    assert np.all(np.isfinite(psi[mask_far]))
    assert np.all(np.isfinite(BR[mask_far]))
    assert np.all(np.isfinite(BZ[mask_far]))

    print("greens.py self-test passed")
