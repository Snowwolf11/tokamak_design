"""
plot_fieldlines.py
==================

Simple 3D magnetic field line visualization for axisymmetric equilibria.

We treat the magnetic field in cylindrical coordinates (R,phi,Z):
  dR/ds   = BR / |B|
  dZ/ds   = BZ / |B|
  dphi/ds = Bphi / (R |B|)

Then convert to Cartesian for plotting:
  x = R cos(phi), y = R sin(phi), z = Z

This gives an "exemplary" 3D picture (great for sanity/debugging).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from .style import set_axes_equal_3d

def trace_fieldline(
    Z: np.ndarray,
    R: np.ndarray,
    BR: np.ndarray,
    BZ: np.ndarray,
    Bphi: np.ndarray,
    *,
    R0: float,
    Z0: float,
    phi0: float = 0.0,
    ds: float = 0.01,
    nsteps: int = 2500,
    stop_if_outside: bool = True,
) -> np.ndarray:
    """
    Trace one field line, returning (N,3) array of Cartesian points.

    ds is a *pseudo arc-length* step; choose based on field strength / grid size.
    """
    Z = np.asarray(Z, float)
    R = np.asarray(R, float)

    BRi = RegularGridInterpolator((Z, R), np.asarray(BR, float), bounds_error=False, fill_value=np.nan)
    BZi = RegularGridInterpolator((Z, R), np.asarray(BZ, float), bounds_error=False, fill_value=np.nan)
    BPi = RegularGridInterpolator((Z, R), np.asarray(Bphi, float), bounds_error=False, fill_value=np.nan)

    Rk = float(R0)
    Zk = float(Z0)
    phik = float(phi0)

    pts = []
    for _ in range(int(nsteps)):
        bR = BRi((Zk, Rk))
        bZ = BZi((Zk, Rk))
        bP = BPi((Zk, Rk))
        if not (np.isfinite(bR) and np.isfinite(bZ) and np.isfinite(bP)):
            break

        Bmag = float(np.sqrt(bR * bR + bZ * bZ + bP * bP))
        if Bmag < 1e-14:
            break

        # fieldline ODE
        dR = float(bR / Bmag) * ds
        dZ = float(bZ / Bmag) * ds
        dphi = float(bP / (max(Rk, 1e-9) * Bmag)) * ds

        Rk += dR
        Zk += dZ
        phik += dphi

        if stop_if_outside:
            if (Rk < R[0]) or (Rk > R[-1]) or (Zk < Z[0]) or (Zk > Z[-1]):
                break

        x = Rk * np.cos(phik)
        y = Rk * np.sin(phik)
        pts.append((x, y, Zk))

    return np.asarray(pts, float)


def plot_fieldlines_3d(
    *,
    run_id: str,
    schema_version: str,
    R: np.ndarray,
    Z: np.ndarray,
    BR: np.ndarray,
    BZ: np.ndarray,
    Bphi: np.ndarray,
    seeds: Sequence[Tuple[float, float]],
    ds: float = 0.01,
    nsteps: int = 2500,
) -> plt.Figure:
    """
    Plot a few field lines in 3D from given (R,Z) seed points.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Run: {run_id}   (schema {schema_version})\nExemplary 3D magnetic field lines")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    any_pts = False
    for (Rs, Zs) in seeds:
        pts = trace_fieldline(Z, R, BR, BZ, Bphi, R0=Rs, Z0=Zs, ds=ds, nsteps=nsteps)
        if pts.shape[0] > 5:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.2)
            any_pts = True
    # If nothing plotted, keep default limits; otherwise enforce equal scaling
    if any_pts:
        set_axes_equal_3d(ax)
        
    return fig
