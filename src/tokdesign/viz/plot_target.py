"""
plot_target.py
==============

Overlay target LCFS boundary on device geometry.

Pure plotting: expects arrays already loaded.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .style import default_fig_ax, set_axes_equal, run_header
from .plot_device import plot_device_geometry


def plot_target_overlay(
    *,
    run_id: str,
    schema_version: str,
    vessel_boundary: Optional[np.ndarray],
    coil_names: Optional[Sequence[str]],
    coil_centers: Optional[np.ndarray],
    coil_radii: Optional[np.ndarray],
    target_boundary: Optional[np.ndarray],
    psi_lcfs: Optional[float] = None,
    annotate_shape: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot device + target LCFS.

    Returns
    -------
    fig, ax
    """
    fig, ax = default_fig_ax(figsize=(7, 7))
    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle="Target overlay")

    # Reuse device plotting on same axis
    plot_device_geometry(
        run_id=run_id,
        schema_version=schema_version,
        vessel_boundary=vessel_boundary,
        coil_names=coil_names,
        coil_centers=coil_centers,
        coil_radii=coil_radii,
        ax=ax,
    )

    if target_boundary is not None:
        ax.plot(target_boundary[:, 0], target_boundary[:, 1], lw=2.0, label="target LCFS")
        ax.legend(loc="best")

        if annotate_shape:
            try:
                from tokdesign.geometry._plasma_boundary import boundary_kappa_delta, boundary_area
                kappa, delta = boundary_kappa_delta(target_boundary)
                area = boundary_area(target_boundary)
                txt = f"κ={kappa:.3f}, δ={delta:.3f}, A={area:.3f} m²"
                if psi_lcfs is not None:
                    txt += f", ψ_lcfs={psi_lcfs:g}"
                ax.text(
                    0.02, 0.02, txt,
                    transform=ax.transAxes,
                    fontsize=9,
                    va="bottom",
                    ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
                )
            except Exception:
                pass

    set_axes_equal(ax)
    return fig, ax
