"""
plot_coil_fit.py
================

Visualization utilities for stage 04 (PF coil current fit).

Includes:
• Boundary ψ vs poloidal angle, with mean line
• Boundary Δψ (cyclic first difference), with contour_rms annotation
• Currents vs limits (mark clamped)
• Vacuum flux surfaces (ψ_vac from G_psi and fitted I_pf)
  - vessel boundary + target boundary overlay
  - optional faint overlay of GS fixed-boundary ψ contours (if available)

Conventions
-----------
• Grid arrays:
    R: (NR,)
    Z: (NZ,)
    fields/psi: (NZ, NR)
• Greens:
    G_psi: (Nc, NZ, NR) with coil index first
• Boundary points:
    boundary_pts: (Nb, 2) columns [R, Z]

Notes
-----
The “poloidal angle” used here is a geometric parameterization around the
boundary centroid. It assumes boundary_pts are ordered around the contour.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from tokdesign.viz.style import default_fig_ax, run_header, set_axes_equal


# ============================================================
# Small helpers
# ============================================================

def _poloidal_angle(boundary_pts: np.ndarray) -> np.ndarray:
    """
    Compute a robust poloidal angle parameter for boundary points.

    Uses centroid as reference:
        θ = atan2(Z - Zc, R - Rc) mapped to [0, 2π).
    """
    boundary_pts = np.asarray(boundary_pts, float)
    Rc = float(np.mean(boundary_pts[:, 0]))
    Zc = float(np.mean(boundary_pts[:, 1]))
    th = np.arctan2(boundary_pts[:, 1] - Zc, boundary_pts[:, 0] - Rc)
    return np.mod(th, 2.0 * np.pi)


def _cyclic_dpsi(psi: np.ndarray) -> np.ndarray:
    """Cyclic first difference: dpsi[k] = psi[k+1] - psi[k] with wrap."""
    psi = np.asarray(psi, float).reshape(-1)
    return np.roll(psi, -1) - psi


def compute_psi_vacuum(G_psi: np.ndarray, I_pf: np.ndarray) -> np.ndarray:
    """
    Vacuum flux on grid from coil greens and currents:
        psi_vac[Z,R] = sum_c G_psi[c,Z,R] * I_pf[c]
    """
    G_psi = np.asarray(G_psi, float)
    I_pf = np.asarray(I_pf, float).reshape(-1)
    if G_psi.ndim != 3:
        raise ValueError("G_psi must have shape (Nc, NZ, NR).")
    Nc = G_psi.shape[0]
    if I_pf.size != Nc:
        raise ValueError(f"I_pf size mismatch: expected Nc={Nc}, got {I_pf.size}")
    return np.tensordot(I_pf, G_psi, axes=(0, 0)).astype(float)  # (NZ, NR)


# ============================================================
# Plots: boundary diagnostics
# ============================================================

def plot_boundary_psi_vs_angle(
    *,
    run_id: str,
    schema_version: str,
    boundary_pts: np.ndarray,
    psi_boundary_fit: np.ndarray,
    mean_line: bool = True,
    subtitle: str = "Coil-fit boundary ψ",
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=subtitle)

    th = _poloidal_angle(boundary_pts)
    psi = np.asarray(psi_boundary_fit, float).reshape(-1)

    order = np.argsort(th)
    ths = th[order]
    ps = psi[order]

    ax.plot(ths, ps, lw=1.8, label="ψ_fit on boundary")
    mu = float(np.mean(psi))
    if mean_line:
        ax.axhline(mu, lw=1.2, ls="--", label=f"mean={mu:.5g}")

    ax.set_xlabel("poloidal angle θ [rad]")
    ax.set_ylabel("ψ [Wb/rad]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig, ax


def plot_boundary_dpsi(
    *,
    run_id: str,
    schema_version: str,
    boundary_pts: np.ndarray,
    psi_boundary_fit: np.ndarray,
    contour_rms: Optional[float] = None,
    subtitle: str = "Coil-fit Δψ along boundary",
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=subtitle)

    th = _poloidal_angle(boundary_pts)
    psi = np.asarray(psi_boundary_fit, float).reshape(-1)
    dpsi = _cyclic_dpsi(psi)

    order = np.argsort(th)
    ths = th[order]
    dps = dpsi[order]

    ax.plot(ths, dps, lw=1.6, label="Δψ = ψ[k+1]-ψ[k]")
    ax.axhline(0.0, lw=1.0, ls="--")

    rms = float(np.sqrt(np.mean(dpsi * dpsi)))
    rep = float(rms if contour_rms is None else contour_rms)

    ax.text(
        0.02, 0.98,
        f"RMS(Δψ)={rms:.4g}\nreported contour_rms={rep:.4g}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )

    ax.set_xlabel("poloidal angle θ [rad]")
    ax.set_ylabel("Δψ [Wb/rad]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return fig, ax


def plot_currents_vs_limits(
    *,
    run_id: str,
    schema_version: str,
    coil_names: Optional[Sequence[str]],
    I_pf: np.ndarray,
    I_max: np.ndarray,
    clamped: Optional[np.ndarray] = None,
    subtitle: str = "PF currents vs limits",
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=subtitle)

    I_pf = np.asarray(I_pf, float).reshape(-1)
    I_max = np.asarray(I_max, float).reshape(-1)
    Nc = I_pf.size
    if I_max.size != Nc:
        raise ValueError("I_max size mismatch with I_pf.")

    labels = [f"C{i}" for i in range(Nc)] if coil_names is None else [str(x) for x in coil_names]

    x = np.arange(Nc)
    ax.bar(x, I_pf)

    ax.plot(x, I_max, lw=1.6, ls="--", label="+I_max")
    ax.plot(x, -I_max, lw=1.6, ls="--", label="-I_max")

    if clamped is not None:
        clamped = np.asarray(clamped).reshape(-1).astype(bool)
        if clamped.size == Nc and np.any(clamped):
            idx = np.where(clamped)[0]
            ax.scatter(idx, I_pf[idx], s=70, marker="x", label="clamped")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Current [A]")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    nclamp = int(np.sum(clamped)) if clamped is not None and np.size(clamped) == Nc else 0
    ax.text(
        0.98, 0.98,
        f"Nc={Nc}, clamped={nclamp}",
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
    )
    return fig, ax


# ============================================================
# Plots: vacuum flux surfaces from 04 currents
# ============================================================

def plot_vacuum_flux_surfaces_compare(
    *,
    run_id: str,
    schema_version: str,
    R: np.ndarray,
    Z: np.ndarray,
    psi_vac: np.ndarray,
    vessel_boundary: Optional[np.ndarray] = None,
    target_boundary: Optional[np.ndarray] = None,
    psi_boundary_ref: Optional[float] = None,
    # Optional GS overlay (fixed-boundary)
    psi_gs: Optional[np.ndarray] = None,
    psi_gs_axis: Optional[float] = None,
    psi_gs_lcfs: Optional[float] = None,
    subtitle: str = "Vacuum flux surfaces (coils) + GS comparison",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Contour plot of ψ_vac with overlays:
      • vessel boundary
      • target LCFS
      • optional faint GS contours (if available)
    """
    R = np.asarray(R, float).reshape(-1)
    Z = np.asarray(Z, float).reshape(-1)
    psi_vac = np.asarray(psi_vac, float)

    fig, ax = default_fig_ax(figsize=(7.8, 7.8))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=subtitle)

    # Choose vacuum levels (sensible defaults)
    if psi_boundary_ref is not None and np.isfinite(psi_boundary_ref):
        c = float(psi_boundary_ref)
        span = 0.25 * (float(np.nanmax(psi_vac)) - float(np.nanmin(psi_vac)))
        span = max(span, 1e-12)
        levels = np.linspace(c - span, c + span, 18)
    else:
        lo, hi = np.nanpercentile(psi_vac, [5.0, 95.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
            lo, hi = float(np.nanmin(psi_vac)), float(np.nanmax(psi_vac))
        levels = np.linspace(lo, hi, 18)

    cs = ax.contour(R, Z, psi_vac, levels=levels, linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2g")

    if psi_boundary_ref is not None and np.isfinite(psi_boundary_ref):
        ax.contour(R, Z, psi_vac, levels=[float(psi_boundary_ref)], linewidths=2.2)

    if vessel_boundary is not None:
        vb = np.asarray(vessel_boundary, float)
        ax.plot(vb[:, 0], vb[:, 1], lw=2.2, label="vessel")

    if target_boundary is not None:
        tb = np.asarray(target_boundary, float)
        ax.plot(tb[:, 0], tb[:, 1], lw=2.2, label="target LCFS")

    # Faint GS overlay for comparison (if present)
    if psi_gs is not None and psi_gs_axis is not None and psi_gs_lcfs is not None:
        psi_gs = np.asarray(psi_gs, float)
        pa = float(psi_gs_axis)
        pl = float(psi_gs_lcfs)
        frac = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
        levels_gs = pa + frac * (pl - pa)
        ax.contour(R, Z, psi_gs, levels=levels_gs, linewidths=1.0, alpha=0.35)
        ax.text(
            0.02, 0.98,
            "GS fixed-boundary contours (faint)",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.12),
        )

    ax.legend(loc="best")
    set_axes_equal(ax)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return fig, ax


def plot_vacuum_psi_map(
    *,
    run_id: str,
    schema_version: str,
    R: np.ndarray,
    Z: np.ndarray,
    psi_vac: np.ndarray,
    vessel_boundary: Optional[np.ndarray] = None,
    target_boundary: Optional[np.ndarray] = None,
    subtitle: str = "Vacuum ψ map (coils)",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of ψ_vac with vessel/target overlays (useful for debugging scale/sign).
    """
    R = np.asarray(R, float).reshape(-1)
    Z = np.asarray(Z, float).reshape(-1)
    psi_vac = np.asarray(psi_vac, float)

    fig, ax = default_fig_ax(figsize=(7.8, 7.0))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=subtitle)

    # pcolormesh expects edges; but for uniform grid this is fine for debug
    im = ax.pcolormesh(R, Z, psi_vac, shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ψ_vac [Wb/rad]")

    if vessel_boundary is not None:
        vb = np.asarray(vessel_boundary, float)
        ax.plot(vb[:, 0], vb[:, 1], lw=2.2, label="vessel")

    if target_boundary is not None:
        tb = np.asarray(target_boundary, float)
        ax.plot(tb[:, 0], tb[:, 1], lw=2.2, label="target LCFS")

    ax.legend(loc="best")
    set_axes_equal(ax)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return fig, ax