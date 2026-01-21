"""
plot_equilibrium.py
===================

Equilibrium plotting utilities.

Plots provided (pure plotting; no HDF5 reads):
- Poloidal flux psi(R,Z): heatmap + contours + LCFS overlay
- Generic 2D scalar maps on the poloidal cross-section (p, F, jphi, |B|, Bp, etc.)
- Midplane cuts (Z≈0): psi(R), BR(R), BZ(R), Bphi(R), etc.

Conventions
-----------
- R: (NR,)
- Z: (NZ,)
- 2D arrays: (NZ, NR) indexing [iz, ir]
- boundaries: (N,2) arrays with columns (R,Z), preferably closed
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .style import default_fig_ax, set_axes_equal, run_header


def _closest_index(arr: np.ndarray, value: float) -> int:
    arr = np.asarray(arr, float)
    return int(np.argmin(np.abs(arr - float(value))))


def _ensure_closed(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("poly must have shape (N,2)")
    if poly.shape[0] < 3:
        return poly
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def plot_psi_map(
    *,
    run_id: str,
    schema_version: str,
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    psi_axis: Optional[float] = None,
    psi_lcfs: Optional[float] = None,
    vessel_boundary: Optional[np.ndarray] = None,
    lcfs_boundary: Optional[np.ndarray] = None,
    extra_flux_levels: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    n_contours: int = 40,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Poloidal flux map: heatmap + contours + optional LCFS and vessel overlay.

    extra_flux_levels are in normalized flux psi_n (0..1) if psi_axis+psi_lcfs are provided.
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    psi = np.asarray(psi, float)

    fig, ax = default_fig_ax(figsize=(8, 7))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle="Equilibrium: poloidal flux ψ(R,Z)")

    # Heatmap
    im = ax.imshow(
        psi,
        origin="lower",
        extent=[R[0], R[-1], Z[0], Z[-1]],
        aspect="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("ψ [Wb/rad]")

    # Contours of psi
    cs = ax.contour(R, Z, psi, levels=n_contours, linewidths=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

    # Optional LCFS polyline
    if lcfs_boundary is not None:
        lcfs = _ensure_closed(lcfs_boundary)
        ax.plot(lcfs[:, 0], lcfs[:, 1], lw=2.2, label="LCFS (target)")

    # Optional vessel
    if vessel_boundary is not None:
        vb = _ensure_closed(vessel_boundary)
        ax.plot(vb[:, 0], vb[:, 1], lw=2.2, label="vessel")

    # Optional normalized-flux contours
    if (psi_axis is not None) and (psi_lcfs is not None) and np.isfinite(psi_axis) and np.isfinite(psi_lcfs):
        # psi_n = (psi - psi_axis)/(psi_lcfs - psi_axis)
        denom = (psi_lcfs - psi_axis)
        if abs(denom) > 0:
            psin = (psi - psi_axis) / denom
            levels = [float(x) for x in extra_flux_levels if 0.0 < x < 1.0]
            if levels:
                ax.contour(R, Z, psin, levels=levels, linewidths=1.3, linestyles="--")

    ax.legend(loc="best")
    set_axes_equal(ax)
    return fig, ax


def plot_scalar_map(
    *,
    run_id: str,
    schema_version: str,
    title: str,
    R: np.ndarray,
    Z: np.ndarray,
    field: np.ndarray,
    cbar_label: str,
    vessel_boundary: Optional[np.ndarray] = None,
    lcfs_boundary: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generic 2D scalar map plot on poloidal plane.
    If mask is provided, plots masked-out region as NaN (transparent in imshow).
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    field = np.asarray(field, float)

    fig, ax = default_fig_ax(figsize=(8, 7))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=title)

    fplot = field.copy()
    if mask is not None:
        mask = np.asarray(mask, bool)
        if mask.shape == fplot.shape:
            fplot[~mask] = np.nan

    im = ax.imshow(
        fplot,
        origin="lower",
        extent=[R[0], R[-1], Z[0], Z[-1]],
        aspect="auto",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)

    if lcfs_boundary is not None:
        lcfs = _ensure_closed(lcfs_boundary)
        ax.plot(lcfs[:, 0], lcfs[:, 1], lw=2.0, label="LCFS (target)")
    if vessel_boundary is not None:
        vb = _ensure_closed(vessel_boundary)
        ax.plot(vb[:, 0], vb[:, 1], lw=2.0, label="vessel")

    ax.legend(loc="best")
    set_axes_equal(ax)
    return fig, ax


def plot_midplane_cuts(
    *,
    run_id: str,
    schema_version: str,
    R: np.ndarray,
    Z: np.ndarray,
    psi: Optional[np.ndarray] = None,
    BR: Optional[np.ndarray] = None,
    BZ: Optional[np.ndarray] = None,
    Bphi: Optional[np.ndarray] = None,
    jphi: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    z0: float = 0.0,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot several midplane (Z≈z0) cuts vs R on a single axis with multiple y-scales avoided:
    we use separate subplots stacked vertically.
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    iz = _closest_index(Z, z0)

    # Build list of (name, values, ylabel)
    series = []
    if psi is not None:
        series.append(("psi", np.asarray(psi, float)[iz, :], "ψ [Wb/rad]"))
    if p is not None:
        series.append(("p", np.asarray(p, float)[iz, :], "p [Pa]"))
    if jphi is not None:
        series.append(("jphi", np.asarray(jphi, float)[iz, :], "jφ [A/m²]"))
    if BR is not None:
        series.append(("BR", np.asarray(BR, float)[iz, :], "BR [T]"))
    if BZ is not None:
        series.append(("BZ", np.asarray(BZ, float)[iz, :], "BZ [T]"))
    if Bphi is not None:
        series.append(("Bphi", np.asarray(Bphi, float)[iz, :], "Bφ [T]"))

    n = max(1, len(series))
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    axes[0].set_title(f"Run: {run_id}   (schema {schema_version})\nMidplane cuts at Z≈{Z[iz]:.3f} m")

    for ax, (name, y, ylabel) in zip(axes, series):
        ax.plot(R, y, lw=1.8, label=name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("R [m]")
    return fig, axes[0]
