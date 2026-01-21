# src/tokdesign/viz/plot_metrics.py
"""
plot_metrics.py
===============

Small plotting helpers for *scalar / 1D* metrics that show up often in reports and quicklook.

Design (same as other tokdesign.viz modules)
-------------------------------------------
• Pure plotting: takes arrays + metadata, returns matplotlib Figure/Axes.
• No HDF5 access here.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from .style import run_header


def plot_radial_profiles(
    *,
    run_id: str,
    schema_version: str,
    rho: np.ndarray,
    p: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
    q: Optional[np.ndarray] = None,
    s: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot 1D radial profiles (already sampled on a flux label rho).

    Parameters
    ----------
    rho : (N,)
        Normalized radius (or other 1D flux label) in [0..1]
    p, F, q, s, alpha : (N,) or None
        Any subset of profiles. Missing ones are skipped.

    Returns
    -------
    fig, axes
    """
    rho = np.asarray(rho, float).ravel()

    items = []
    if p is not None:
        items.append(("p", np.asarray(p, float).ravel(), "p [Pa]"))
    if F is not None:
        items.append(("F", np.asarray(F, float).ravel(), "F [T·m]"))
    if q is not None:
        items.append(("q", np.asarray(q, float).ravel(), "q [-]"))
    if s is not None:
        items.append(("s", np.asarray(s, float).ravel(), "s [-]"))
    if alpha is not None:
        items.append(("alpha", np.asarray(alpha, float).ravel(), "alpha [-]"))

    n = max(1, len(items))
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    run_header(
        axes[0],
        run_id=run_id,
        schema_version=schema_version,
        subtitle="1D profiles vs rho (best candidate)",
    )

    if not items:
        axes[0].text(0.02, 0.5, "No 1D profiles provided.", transform=axes[0].transAxes)
        axes[0].set_xlabel("rho")
        axes[0].grid(True, alpha=0.25)
        return fig, axes

    for ax, (name, y, ylabel) in zip(axes, items):
        m = np.isfinite(rho) & np.isfinite(y)
        ax.plot(rho[m], y[m], lw=2.0, label=name)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("rho")
    return fig, axes
