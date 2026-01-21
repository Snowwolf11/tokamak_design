"""
tokdesign.geometry.grid
======================

Grid generation utilities for tokdesign.

This module is intentionally *not* tied to the Grad–Shafranov equation.
It generates a reusable axisymmetric (R,Z) mesh that many physics modules can use.

Stage 01 expects a grid object with:
  - R : 1D array (nR,)
  - Z : 1D array (nZ,)
  - RR: 2D mesh (nZ, nR)
  - ZZ: 2D mesh (nZ, nR)
  - dR: scalar
  - dZ: scalar

Supported config shapes
-----------------------
We read grid settings from cfg_opt["numerics"]["grid"] by convention.

Examples:

numerics:
  grid:
    type: rect
    R: {min: 0.2, max: 3.0, n: 241}
    Z: {min: -2.0, max: 2.0, n: 321}

or (also accepted):
numerics:
  grid:
    type: rect
    R_min: 0.2
    R_max: 3.0
    nR: 241
    Z_min: -2.0
    Z_max: 2.0
    nZ: 321
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


# -----------------------------------------------------------------------------
# Public grid container
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RZGrid:
    """
    Simple immutable container for an (R,Z) mesh.

    Shapes
    ------
    R:  (nR,)
    Z:  (nZ,)
    RR: (nZ,nR)
    ZZ: (nZ,nR)
    """
    R: np.ndarray
    Z: np.ndarray
    RR: np.ndarray
    ZZ: np.ndarray
    dR: float
    dZ: float

    @property
    def nR(self) -> int:
        return int(self.R.shape[0])

    @property
    def nZ(self) -> int:
        return int(self.Z.shape[0])


# -----------------------------------------------------------------------------
# Main builder entry point
# -----------------------------------------------------------------------------

def build_grid(
    cfg_opt: Dict[str, Any],
    cfg_space: Optional[Dict[str, Any]] = None,
) -> RZGrid:
    """
    Build an RZGrid from config dicts.

    Parameters
    ----------
    cfg_opt:
      Usually equilibrium_optimization.yaml (already loaded as dict).
      We read cfg_opt["numerics"]["grid"].

    cfg_space:
      Unused for now, but kept in signature for future:
        - auto-expanding domain based on expected LCFS size
        - using equilibrium_space conventions for default bounds
      Stage01 calls build_grid(cfg_opt, cfg_space) so we accept it.

    Returns
    -------
    RZGrid
    """
    grid_cfg = _get_grid_cfg(cfg_opt)

    gtype = str(grid_cfg.get("type", "rect")).strip().lower()
    if gtype in ("rect", "rectilinear", "cartesian", "rz_rect"):
        return _build_rect_grid(grid_cfg)

    raise ValueError(f"Unsupported grid type: {grid_cfg.get('type')!r}. Supported: 'rect'.")


# -----------------------------------------------------------------------------
# Config parsing helpers
# -----------------------------------------------------------------------------

def _get_grid_cfg(cfg_opt: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cfg_opt, dict):
        raise TypeError("cfg_opt must be a dict.")
    numerics = cfg_opt.get("numerics", {})
    if not isinstance(numerics, dict):
        raise TypeError("cfg_opt['numerics'] must be a dict.")
    grid_cfg = numerics.get("grid", {})
    if not isinstance(grid_cfg, dict):
        raise TypeError("cfg_opt['numerics']['grid'] must be a dict.")
    return grid_cfg


def _read_axis_cfg(grid_cfg: Dict[str, Any], axis: str) -> Tuple[float, float, int]:
    """
    Read axis config for R or Z.

    Accepts:
      grid_cfg["R"] = {"min":..., "max":..., "n":...}
    or:
      grid_cfg["R_min"], grid_cfg["R_max"], grid_cfg["nR"]

    Returns:
      (amin, amax, n)
    """
    axis = axis.upper()
    if axis not in ("R", "Z"):
        raise ValueError("axis must be 'R' or 'Z'.")

    # Preferred nested form: R: {min,max,n}
    nested = grid_cfg.get(axis, None)
    if isinstance(nested, dict):
        amin = _as_float(nested.get("min", None), f"numerics.grid.{axis}.min")
        amax = _as_float(nested.get("max", None), f"numerics.grid.{axis}.max")
        n = _as_int(nested.get("n", None), f"numerics.grid.{axis}.n")
        return amin, amax, n

    # Flat keys fallback
    amin = _as_float(grid_cfg.get(f"{axis}_min", None), f"numerics.grid.{axis}_min")
    amax = _as_float(grid_cfg.get(f"{axis}_max", None), f"numerics.grid.{axis}_max")
    n = _as_int(grid_cfg.get(f"n{axis}", None), f"numerics.grid.n{axis}")
    return amin, amax, n


def _as_float(x: Any, name: str) -> float:
    if x is None:
        raise ValueError(f"Missing required config value: {name}")
    try:
        return float(x)
    except Exception as e:
        raise ValueError(f"Invalid float for {name}: {x!r}") from e


def _as_int(x: Any, name: str) -> int:
    if x is None:
        raise ValueError(f"Missing required config value: {name}")
    try:
        xi = int(x)
    except Exception as e:
        raise ValueError(f"Invalid int for {name}: {x!r}") from e
    if xi < 2:
        raise ValueError(f"{name} must be >= 2 (got {xi})")
    return xi


# -----------------------------------------------------------------------------
# Rectilinear grid builder
# -----------------------------------------------------------------------------

def _build_rect_grid(grid_cfg: Dict[str, Any]) -> RZGrid:
    """
    Build a rectilinear (R,Z) grid with meshgrid.

    Conventions
    -----------
    We use numpy.meshgrid with indexing='xy', but we explicitly return RR,ZZ
    shaped (nZ,nR) to match your schema and typical GS solver memory layout
    (Z index first, then R).

    R = linspace(Rmin, Rmax, nR)
    Z = linspace(Zmin, Zmax, nZ)
    RR, ZZ = meshgrid(R, Z)  -> shapes (nZ,nR)
    """
    Rmin, Rmax, nR = _read_axis_cfg(grid_cfg, "R")
    Zmin, Zmax, nZ = _read_axis_cfg(grid_cfg, "Z")

    if not (np.isfinite(Rmin) and np.isfinite(Rmax) and Rmax > Rmin):
        raise ValueError(f"Invalid R range: min={Rmin}, max={Rmax}")
    if not (np.isfinite(Zmin) and np.isfinite(Zmax) and Zmax > Zmin):
        raise ValueError(f"Invalid Z range: min={Zmin}, max={Zmax}")

    R = np.linspace(Rmin, Rmax, nR, dtype=float)
    Z = np.linspace(Zmin, Zmax, nZ, dtype=float)

    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    # Mesh: RR and ZZ are (nZ,nR)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    return RZGrid(R=R, Z=Z, RR=RR, ZZ=ZZ, dR=dR, dZ=dZ)
