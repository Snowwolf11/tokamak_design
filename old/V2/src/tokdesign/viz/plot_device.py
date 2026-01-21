"""
plot_device.py
==============

Device geometry plotting:
• vessel boundary
• PF coil proxies (circles with radius a)
• coil labels

Pure plotting: no HDF5 reads here.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .style import default_fig_ax, set_axes_equal, run_header


def plot_device_geometry(
    *,
    run_id: str,
    schema_version: str,
    vessel_boundary: Optional[np.ndarray],
    coil_names: Optional[Sequence[str]],
    coil_centers: Optional[np.ndarray],
    coil_radii: Optional[np.ndarray],
    coil_currents: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot vessel and coils.

    Parameters
    ----------
    vessel_boundary : (N,2) array or None
    coil_centers : (Nc,2) array or None
    coil_radii : (Nc,) array or None
    coil_names : list[str] or None
    coil_currents : (Nc,) array or None
        If provided, annotates currents next to coil names (optional).

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = default_fig_ax(figsize=(7, 7))
    else:
        fig = ax.figure

    run_header(ax, run_id=run_id, schema_version=schema_version, subtitle="Device geometry")

    # Vessel
    if vessel_boundary is not None:
        ax.plot(vessel_boundary[:, 0], vessel_boundary[:, 1], lw=2.5, label="vessel")

    # Coils
    if coil_centers is not None and coil_radii is not None and coil_names is not None:
        for i, (name, (Rc, Zc), a) in enumerate(zip(coil_names, coil_centers, coil_radii)):
            circ = plt.Circle((Rc, Zc), float(a), fill=False, linewidth=1.6)
            ax.add_patch(circ)
            ax.plot([Rc], [Zc], marker="x")

            label = f"{name}"
            if coil_currents is not None and i < len(coil_currents):
                label += f"  ({coil_currents[i]:.2e} A)"

            ax.text(Rc, Zc, " " + label, va="center", fontsize=9)

        ax.legend(loc="best")

    set_axes_equal(ax)
    return fig, ax
