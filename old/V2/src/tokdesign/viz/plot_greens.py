"""
plot_greens.py
==============

Plot coil psi Green's functions (psi_per_amp) as contour maps.

Pure plotting: accepts grid arrays and G_psi already loaded.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .style import default_fig_ax, set_axes_equal, run_header


def plot_coil_greens(
    *,
    run_id: str,
    schema_version: str,
    R: np.ndarray,
    Z: np.ndarray,
    vessel_boundary: Optional[np.ndarray],
    coil_names: Sequence[str],
    coil_centers: Optional[np.ndarray],
    G_psi: np.ndarray,  # (Nc, NZ, NR)
    which: Optional[Sequence[str]] = None,
    max_plots: int = 4,
    n_levels: int = 30,
) -> List[Tuple[str, plt.Figure, plt.Axes]]:
    """
    Create contour plots of psi-per-amp for selected coils.

    Returns
    -------
    list of (coil_name, fig, ax)
    """
    if G_psi.ndim != 3:
        raise ValueError(f"G_psi must have shape (Nc,NZ,NR), got {G_psi.shape}")

    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    name_to_idx = {n: i for i, n in enumerate(coil_names)}

    if which is None or len(which) == 0:
        indices = list(range(len(coil_names)))
    else:
        indices = []
        for n in which:
            if n not in name_to_idx:
                raise ValueError(f"Requested coil '{n}' not found. Available: {list(coil_names)}")
            indices.append(name_to_idx[n])

    indices = indices[:max_plots]

    out: List[Tuple[str, plt.Figure, plt.Axes]] = []
    for idx in indices:
        cname = coil_names[idx]
        G = G_psi[idx]

        fig, ax = default_fig_ax(figsize=(7, 7))
        run_header(ax, run_id=run_id, schema_version=schema_version, subtitle=f"Green's function — {cname}")

        cs = ax.contour(RR, ZZ, G, levels=n_levels)
        ax.clabel(cs, inline=True, fontsize=7)

        if vessel_boundary is not None:
            ax.plot(vessel_boundary[:, 0], vessel_boundary[:, 1], lw=2.0)

        if coil_centers is not None:
            Rc, Zc = coil_centers[idx]
            ax.plot([Rc], [Zc], marker="o")
            ax.text(Rc, Zc, f" {cname}", va="center", fontsize=9)

        set_axes_equal(ax)
        out.append((cname, fig, ax))

    return out
