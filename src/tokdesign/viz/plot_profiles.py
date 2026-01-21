"""
plot_profiles.py
================

1D profile plots derived from 2D equilibrium fields, shown as functions of normalized flux ψ̄.

We compute ψ̄ on the grid, then bin data from the plasma region (mask) to estimate:
  - mean and spread of p(ψ̄), F(ψ̄), jφ(ψ̄), |B|(ψ̄), Bp(ψ̄)

This is very useful for debugging:
  - profile model issues
  - normalization issues (psi_axis/psi_lcfs swapped)
  - mask problems (plasma region wrong)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .style import run_header


def _binned_stats(x: np.ndarray, y: np.ndarray, nbins: int = 40) -> Dict[str, np.ndarray]:
    """
    Return bin centers and mean/std of y in bins of x.
    x must be in [0,1] approximately; we clip to [0,1].
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()

    m = np.isfinite(x) & np.isfinite(y)
    x = np.clip(x[m], 0.0, 1.0)
    y = y[m]

    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.digitize(x, edges) - 1

    xc = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(nbins, np.nan)
    std = np.full(nbins, np.nan)
    count = np.zeros(nbins, dtype=int)

    for b in range(nbins):
        mb = idx == b
        if np.any(mb):
            vals = y[mb]
            mean[b] = float(np.mean(vals))
            std[b] = float(np.std(vals))
            count[b] = int(vals.size)

    return {"xc": xc, "mean": mean, "std": std, "count": count}


def plot_profiles_vs_psin(
    *,
    run_id: str,
    schema_version: str,
    psin: np.ndarray,
    mask: np.ndarray,
    p: Optional[np.ndarray] = None,
    F: Optional[np.ndarray] = None,
    jphi: Optional[np.ndarray] = None,
    Bp: Optional[np.ndarray] = None,
    Bmag: Optional[np.ndarray] = None,
    nbins: int = 45,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Multi-panel profile plot: binned mean ± std vs ψ̄.
    """
    psin = np.asarray(psin, float)
    mask = np.asarray(mask, bool)

    items = []
    if p is not None:
        items.append(("p(ψ̄)", np.asarray(p, float), "p [Pa]"))
    if F is not None:
        items.append(("F(ψ̄)", np.asarray(F, float), "F [T·m]"))
    if jphi is not None:
        items.append(("jφ(ψ̄)", np.asarray(jphi, float), "jφ [A/m²]"))
    if Bp is not None:
        items.append(("Bp(ψ̄)", np.asarray(Bp, float), "Bp [T]"))
    if Bmag is not None:
        items.append(("|B|(ψ̄)", np.asarray(Bmag, float), "|B| [T]"))

    n = max(1, len(items))
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    run_header(axes[0], run_id=run_id, schema_version=schema_version, subtitle="Profiles vs normalized flux ψ̄ (binned)")

    x = psin[mask]

    for ax, (name, arr, ylabel) in zip(axes, items):
        y = arr[mask]
        st = _binned_stats(x, y, nbins=nbins)
        ax.plot(st["xc"], st["mean"], lw=2.0, label=f"{name} mean")
        ax.fill_between(st["xc"], st["mean"] - st["std"], st["mean"] + st["std"], alpha=0.25, label="±1σ")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("ψ̄")
    return fig, axes[0]
