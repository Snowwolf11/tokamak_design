"""
style.py
========

Central plotting style helpers.

Keep this small and non-opinionated in v1, but provide:
• consistent figure sizing
• common axis formatting
• equal-aspect helper
• run header titles
"""

from __future__ import annotations

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

def apply_mpl_defaults() -> None:
    """
    Apply lightweight defaults. This is deliberately minimal (expand later).
    Call once per plotting session (e.g. at start of quicklook).
    """
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


def set_axes_equal(ax: plt.Axes) -> None:
    """Enforce equal aspect ratio for R-Z geometry plots."""
    ax.set_aspect("equal", adjustable="box")


def set_axes_equal_3d(ax) -> None:
    """
    Set 3D plot axes to equal scale so that spheres/circles look like spheres/circles.

    Matplotlib 3D doesn't support ax.set_aspect("equal") reliably, so we do it manually
    by setting x/y/z limits to the same half-range around their midpoints.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    half = 0.5 * max(x_range, y_range, z_range)
    if not np.isfinite(half) or half <= 0:
        return

    ax.set_xlim3d(x_mid - half, x_mid + half)
    ax.set_ylim3d(y_mid - half, y_mid + half)
    ax.set_zlim3d(z_mid - half, z_mid + half)

def run_header(ax: plt.Axes, *, run_id: str, schema_version: str, subtitle: Optional[str] = None) -> None:
    """Standard title header for plots."""
    title = f"Run: {run_id}   (schema {schema_version})"
    if subtitle:
        title = f"{title}\n{subtitle}"
    ax.set_title(title)


def default_fig_ax(figsize: Tuple[float, float] = (7, 7)):
    """Create a figure+axis with consistent defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    return fig, ax
