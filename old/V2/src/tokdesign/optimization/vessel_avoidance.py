"""
vessel_avoidance.py
===================

Hard vessel-intersection constraint helper for PF coil position optimization.

Purpose
-------
This module provides a *minimal* interface for “coil must not be inside / intersect
the vessel” checks, including a finite coil radius `a` and an extra `clearance`.

It is designed to be called by:
  • tokdesign.optimization.coil_position_opt (outer loop)
  • stage scripts (e.g. 04_fit_pf_currents.py) to pre-check initial feasibility

Key design choices (per project conventions)
--------------------------------------------
• No HDF5 I/O: stage scripts read results.h5 and pass arrays in.
• No logging: stage scripts handle warnings; this module returns violations.
• Reuse existing geometry code: relies on tokdesign.geometry.vessel functions.

Constraint definition (hard, no “modes”)
----------------------------------------
A coil (modeled as a disk of radius r = a + clearance) is considered invalid if:
  1) its center lies inside the vessel polygon, OR
  2) its center is closer than r to the vessel boundary polyline.

This is the conservative “no intersection” rule you want.

Inputs
------
• vessel_boundary: (Nv,2) polyline/polygon in (R,Z), closed or open.
• centers:         (Nc,2) coil centers in (R,Z)
• coil_a:          scalar or (Nc,) coil radius [m]
• clearance:       scalar [m], extra margin beyond a

Outputs
-------
• boolean mask per queried coil and/or list of violating indices.

Self-test
---------
Run:
  python -m tokdesign.optimization.vessel_avoidance
(or execute the file directly) to run a lightweight self-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Reuse existing geometry utilities (avoid code duplication)
# NOTE: adjust import path if your package layout differs.
from tokdesign.geometry.vessel import (
    ensure_closed_polyline,
    point_in_polygon,
    min_distance_to_polyline,
    ellipse_boundary,
)

Array = np.ndarray


@dataclass(frozen=True)
class VesselAvoidanceConfig:
    """Configuration for the vessel avoidance constraint."""
    enabled: bool = False
    clearance: float = 0.0  # [m]


def _as_index_array(movable: Optional[Sequence[int]], Nc: int) -> np.ndarray:
    if movable is None:
        return np.arange(Nc, dtype=int)
    idx = np.asarray(list(movable), dtype=int).reshape(-1)
    if np.any(idx < 0) or np.any(idx >= Nc):
        raise ValueError("movable indices out of range")
    return idx


def _expand_coil_a(coil_a: Union[float, Array], Nc: int) -> np.ndarray:
    a = np.asarray(coil_a, float).reshape(-1)
    if a.size == 1:
        return np.full(Nc, float(a.item()), dtype=float)
    if a.size != Nc:
        raise ValueError(f"coil_a must be scalar or length Nc={Nc}, got {a.size}")
    return a


def vessel_violation_mask(
    *,
    vessel_boundary: Array,
    centers: Array,
    coil_a: Union[float, Array],
    clearance: float,
    movable: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Compute a boolean mask indicating which coils violate the vessel constraint.

    Parameters
    ----------
    vessel_boundary:
        (Nv,2) polyline/polygon in (R,Z). Will be closed if open.

    centers:
        (Nc,2) coil centers in (R,Z).

    coil_a:
        scalar or (Nc,) coil radius [m] (finite extent).

    clearance:
        [m] extra margin beyond coil_a.

    movable:
        Optional subset of coil indices to test. If None, tests all coils.

    Returns
    -------
    bad_mask:
        Boolean array of length len(movable) if movable is provided,
        otherwise length Nc, with True for violating coils.
    """
    vessel_boundary = ensure_closed_polyline(np.asarray(vessel_boundary, float))
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (Nc,2)")

    Nc = centers.shape[0]
    idx = _as_index_array(movable, Nc)
    a = _expand_coil_a(coil_a, Nc)
    clearance = float(clearance)

    pts = centers[idx]  # (M,2)
    inside = point_in_polygon(vessel_boundary, pts)  # (M,)
    dmin = min_distance_to_polyline(vessel_boundary, pts)  # (M,)

    r = a[idx] + clearance
    too_close = dmin < r

    return inside | too_close


def violating_indices(
    *,
    vessel_boundary: Array,
    centers: Array,
    coil_a: Union[float, Array],
    clearance: float,
    movable: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Return a list of violating coil indices.

    If movable is given, returned indices are in the original (global) coil index space.
    """
    centers = np.asarray(centers, float)
    Nc = centers.shape[0]
    idx = _as_index_array(movable, Nc)

    bad = vessel_violation_mask(
        vessel_boundary=vessel_boundary,
        centers=centers,
        coil_a=coil_a,
        clearance=clearance,
        movable=idx,
    )
    return [int(i) for i in idx[bad]]


def any_violation(
    *,
    vessel_boundary: Array,
    centers: Array,
    coil_a: Union[float, Array],
    clearance: float,
    movable: Optional[Sequence[int]] = None,
) -> bool:
    """Convenience boolean: True if any coil violates."""
    bad = vessel_violation_mask(
        vessel_boundary=vessel_boundary,
        centers=centers,
        coil_a=coil_a,
        clearance=clearance,
        movable=movable,
    )
    return bool(np.any(bad))


# ============================================================
# SELF TEST
# ============================================================

def _selftest() -> None:
    print("Running vessel_avoidance.py self-test...")

    # Simple ellipse vessel around (R=1.7, Z=0.0)
    vessel = ellipse_boundary(1.7, 0.0, 1.0, 1.4, n_points=200)  # closed

    # Three coils:
    #  - inside center
    #  - outside far away
    #  - outside but near boundary (should violate with sufficient a+clearance)
    centers = np.array([
        [1.7, 0.0],   # inside
        [3.2, 0.0],   # outside
        [2.68, 0.0],  # near right side of ellipse (approx boundary at R~2.7)
    ], dtype=float)

    coil_a = np.array([0.05, 0.05, 0.05], dtype=float)

    # Case 1: small clearance, coil 0 must violate (inside); coil 1 should pass;
    # coil 2 likely passes if not too close.
    bad1 = violating_indices(
        vessel_boundary=vessel,
        centers=centers,
        coil_a=coil_a,
        clearance=0.0,
    )
    print("bad indices (clearance=0):", bad1)
    assert 0 in bad1, "coil at vessel center must violate"
    assert 1 not in bad1, "far outside coil should not violate"

    # Case 2: add clearance so coil 2 becomes invalid (near boundary)
    bad2 = violating_indices(
        vessel_boundary=vessel,
        centers=centers,
        coil_a=coil_a,
        clearance=0.10,   # make keep-out larger
    )
    print("bad indices (clearance=0.10):", bad2)
    assert 0 in bad2
    assert 1 not in bad2
    assert 2 in bad2, "near-boundary coil should violate when keep-out grows"

    # Subset test
    bad_subset = violating_indices(
        vessel_boundary=vessel,
        centers=centers,
        coil_a=coil_a,
        clearance=0.10,
        movable=[1, 2],
    )
    print("bad indices subset [1,2]:", bad_subset)
    assert bad_subset == [2]

    print("vessel_avoidance.py self-test passed.")


if __name__ == "__main__":
    _selftest()
