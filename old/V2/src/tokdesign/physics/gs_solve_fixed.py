"""
gs_solve_fixed.py
=================

Fixed-boundary Grad–Shafranov (GS) solver on a uniform R–Z grid.

This solves the GS equation *inside a prescribed LCFS boundary* (given as a closed
polyline in (R,Z)), using analytic profiles provided by physics/gs_profiles.py.

Equation
--------
    Δ*ψ = -μ0 R jφ(ψ)

with:
    Δ*ψ = R d/dR ( (1/R) dψ/dR ) + d²ψ/dZ²

We solve on a rectangular grid but only for "active" nodes inside the LCFS.
Dirichlet boundary condition:
    ψ = ψ_lcfs on the boundary ring (inactive nodes adjacent to the active region).

Algorithm (Picard iteration)
----------------------------
1) Build an "inside-LCFS" mask on grid nodes.
2) Build sparse Δ* matrix for active nodes.
3) Picard loop:
   - estimate ψ_axis from current ψ (extremum inside mask)
   - compute jφ(ψ) from profiles (via gs_profiles.jphi_from_psi)
   - build RHS: b = -μ0 R jφ
   - apply Dirichlet shift for ψ_lcfs on boundary ring nodes
   - solve linear system A x = b
   - under-relax into ψ
   - check relative change for convergence

Conventions
-----------
- R: shape (NR,), Z: shape (NZ,)
- RR, ZZ, psi: shape (NZ, NR) with indexing [iz, ir]
- lcfs_poly: shape (N, 2) with columns (R,Z); should be closed (first==last).

Notes
-----
- This is a v1 "good enough" fixed-boundary solve: LCFS is imposed by a mask + ring
  Dirichlet values (ψ_lcfs constant).
- Global targets (e.g., enforcing Ip) are NOT handled here. That belongs in the
  driver or a higher-level solver that rescales profiles / source terms.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla
from matplotlib.path import Path

from tokdesign.physics._gs_operator import build_delta_star_matrix, build_index_map, apply_dirichlet
from tokdesign.physics._gs_profiles import GSProfiles, jphi_from_psi, MU0


# ============================================================
# MASK + RING HELPERS
# ============================================================

def _ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("lcfs_poly must have shape (N,2).")
    if poly.shape[0] < 4:
        raise ValueError("lcfs_poly must have at least 4 points (including closure).")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _mask_inside_polyline(RR: np.ndarray, ZZ: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """
    Point-in-polygon mask on grid nodes using matplotlib.path.Path.
    """
    poly = _ensure_closed_polyline(poly)
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    inside = Path(poly).contains_points(pts)
    return inside.reshape(RR.shape)


def _dirichlet_ring_from_mask(mask_active: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask selecting inactive nodes that are 4-neighbors of any active node.
    This is the "Dirichlet ring" that couples into the interior stencil.
    """
    mask_active = np.asarray(mask_active, dtype=bool)
    NZ, NR = mask_active.shape
    ring = np.zeros_like(mask_active, dtype=bool)

    # Any inactive node adjacent to an active node becomes ring=True.
    for iz in range(NZ):
        for ir in range(NR):
            if mask_active[iz, ir]:
                continue
            # check 4-neighbors for an active cell
            if (ir > 0 and mask_active[iz, ir - 1]) or (ir < NR - 1 and mask_active[iz, ir + 1]) or \
               (iz > 0 and mask_active[iz - 1, ir]) or (iz < NZ - 1 and mask_active[iz + 1, ir]):
                ring[iz, ir] = True
    return ring


def _estimate_psi_axis(psi: np.ndarray, mask_active: np.ndarray, psi_lcfs: float) -> Tuple[float, Tuple[int, int]]:
    """
    Estimate psi_axis as the interior extremum of psi (min or max), chosen relative to psi_lcfs.

    In many tokamak conventions psi_lcfs is set to 0 and psi is negative inside,
    so the axis is the minimum. But we handle either by picking the extremum
    farthest from psi_lcfs.
    """
    vals = psi[mask_active]
    if vals.size == 0:
        raise ValueError("Active mask is empty; cannot estimate psi_axis.")

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    # choose extremum farther from boundary value
    if abs(vmin - psi_lcfs) >= abs(vmax - psi_lcfs):
        # axis = argmin
        flat_idx = np.argmin(np.where(mask_active, psi, np.inf))
        iz, ir = np.unravel_index(flat_idx, psi.shape)
        return float(psi[iz, ir]), (int(iz), int(ir))
    else:
        # axis = argmax
        flat_idx = np.argmax(np.where(mask_active, psi, -np.inf))
        iz, ir = np.unravel_index(flat_idx, psi.shape)
        return float(psi[iz, ir]), (int(iz), int(ir))


def _extract_active_rhs(rhs_full: np.ndarray, rev_map: np.ndarray) -> np.ndarray:
    """
    Gather a full-grid RHS (NZ,NR) into the active vector b of shape (N_active,).
    """
    b = np.empty((rev_map.shape[0],), dtype=float)
    for k in range(rev_map.shape[0]):
        iz, ir = int(rev_map[k, 0]), int(rev_map[k, 1])
        b[k] = float(rhs_full[iz, ir])
    return b


def _scatter_active_to_grid(x: np.ndarray, rev_map: np.ndarray, psi_grid: np.ndarray) -> None:
    """
    Scatter active unknown vector x into psi_grid at positions rev_map.
    """
    for k in range(rev_map.shape[0]):
        iz, ir = int(rev_map[k, 0]), int(rev_map[k, 1])
        psi_grid[iz, ir] = float(x[k])


# ============================================================
# MAIN SOLVER
# ============================================================

def solve_fixed_boundary_gs(
    R: np.ndarray,
    Z: np.ndarray,
    RR: np.ndarray,
    ZZ: np.ndarray,
    lcfs_poly: np.ndarray,
    psi_lcfs: float,
    profiles: GSProfiles,
    solver_cfg: Dict[str, Any],
    psi_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Fixed-boundary GS solve inside lcfs_poly.

    Parameters
    ----------
    R, Z : 1D arrays
        Grid coordinates.
    RR, ZZ : 2D arrays (NZ,NR)
        Meshgrid arrays.
    lcfs_poly : array (N,2)
        Closed LCFS boundary polyline in (R,Z).
    psi_lcfs : float
        Dirichlet boundary value on the LCFS ring.
    profiles : GSProfiles
        Pressure + toroidal field function profile models.
    solver_cfg : dict
        Expected keys (with reasonable defaults):
          - max_picard_iter (int, default 50)
          - tol_rel_change (float, default 1e-6)
          - under_relaxation (float in (0,1], default 0.5)
          - linear_solver ("spsolve"|"cg", default "spsolve")
          - cg_tol (float, default 1e-10)
          - cg_max_iter (int, default 2000)
          - clip_psin (bool, default True)
    psi_init : optional 2D array
        Initial guess for psi on the full grid.

    Returns
    -------
    dict with:
      - psi (NZ,NR)
      - psi_axis (float)
      - axis_RZ ((Raxis, Zaxis))
      - axis_ij ((iz, ir))
      - mask (NZ,NR) active mask inside LCFS
      - ring (NZ,NR) dirichlet ring mask (inactive nodes adjacent to active)
      - jphi (NZ,NR)
      - converged (bool)
      - history (dict of lists)
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    RR = np.asarray(RR, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)

    if RR.shape != ZZ.shape:
        raise ValueError("RR and ZZ must have the same shape (NZ,NR).")
    if RR.shape != (Z.size, R.size):
        raise ValueError(f"RR/ZZ shape must be (NZ,NR)=({Z.size},{R.size}), got {RR.shape}.")

    max_iter = int(solver_cfg.get("max_picard_iter", 50))
    tol = float(solver_cfg.get("tol_rel_change", 1e-6))
    w = float(solver_cfg.get("under_relaxation", 0.5))
    if not (0.0 < w <= 1.0):
        raise ValueError("under_relaxation must be in (0,1].")

    linear_solver = str(solver_cfg.get("linear_solver", "spsolve"))
    cg_tol = float(solver_cfg.get("cg_tol", 1e-10))
    cg_max_iter = int(solver_cfg.get("cg_max_iter", 2000))
    clip_psin = bool(solver_cfg.get("clip_psin", True))

    # -----------------------------
    # Build mask and dirichlet ring
    # -----------------------------
    mask_active = _mask_inside_polyline(RR, ZZ, lcfs_poly)

    # Avoid accidentally including the rectangular grid boundary in active set
    mask_active[0, :] = False
    mask_active[-1, :] = False
    mask_active[:, 0] = False
    mask_active[:, -1] = False

    if not np.any(mask_active):
        raise ValueError("Active region is empty after masking; check LCFS vs grid extent.")

    ring = _dirichlet_ring_from_mask(mask_active)
    if not np.any(ring):
        raise ValueError("Dirichlet ring is empty; mask may be malformed or too close to grid edge.")

    # -----------------------------
    # Initialize psi
    # -----------------------------
    if psi_init is None:
        psi = np.full(RR.shape, float(psi_lcfs), dtype=float)
        # Small interior perturbation to help axis detection on first iteration:
        psi[mask_active] = float(psi_lcfs) - 1e-3
    else:
        psi = np.asarray(psi_init, dtype=float).copy()
        if psi.shape != RR.shape:
            raise ValueError(f"psi_init must have shape {RR.shape}, got {psi.shape}")
        # enforce boundary ring to psi_lcfs for consistency
        psi[ring] = float(psi_lcfs)

    # -----------------------------
    # Build sparse operator once (mask fixed in fixed-boundary solve)
    # -----------------------------
    A = build_delta_star_matrix(R, Z, mask=mask_active)
    idx_map, rev_map = build_index_map(mask_active)

    # Pre-build boundary node list for apply_dirichlet
    ring_nodes = list(map(tuple, np.argwhere(ring)))  # [(iz,ir), ...]
    ring_vals = [float(psi_lcfs)] * len(ring_nodes)

    # -----------------------------
    # Picard iteration
    # -----------------------------
    history = {
        "rel_change": [],
        "psi_axis": [],
        "lin_info": [],
    }

    converged = False
    eps_norm = 1e-30

    for it in range(max_iter):
        psi_old = psi.copy()

        psi_axis, axis_ij = _estimate_psi_axis(psi, mask_active, float(psi_lcfs))
        iz_ax, ir_ax = axis_ij
        axis_RZ = (float(R[ir_ax]), float(Z[iz_ax]))

        # Compute jphi everywhere (solver will use it only inside mask)
        jphi = jphi_from_psi(
            psi=psi,
            RR=RR,
            psi_axis=psi_axis,
            psi_lcfs=float(psi_lcfs),
            profiles=profiles,
            clip_psin=clip_psin,
            mu0=MU0,
        )

        # RHS: Δ*ψ = -μ0 R jphi  =>  b = -μ0 * R * jphi
        rhs_full = -MU0 * RR * jphi

        # Gather RHS for active nodes
        b = _extract_active_rhs(rhs_full, rev_map)

        # Shift Dirichlet boundary contributions (ψ=ψ_lcfs on ring nodes)
        b = apply_dirichlet(b, idx_map, ring_nodes, ring_vals, R=R, Z=Z)

        # Solve linear system
        if linear_solver == "spsolve":
            x = spla.spsolve(A, b)
            lin_info = 0
        elif linear_solver == "cg":
            x, lin_info = spla.cg(A, b, tol=cg_tol, maxiter=cg_max_iter)
            if lin_info < 0:
                raise RuntimeError("CG failed with illegal input or breakdown (info < 0).")
        else:
            raise ValueError(f"Unknown linear_solver='{linear_solver}' (use 'spsolve' or 'cg').")

        # Scatter solution into a new psi grid (keep outside values as-is)
        psi_sol = psi.copy()
        _scatter_active_to_grid(x, rev_map, psi_sol)

        # Enforce ring boundary value explicitly
        psi_sol[ring] = float(psi_lcfs)

        # Under-relaxation on active region only
        psi[mask_active] = (1.0 - w) * psi_old[mask_active] + w * psi_sol[mask_active]
        psi[ring] = float(psi_lcfs)

        # Convergence check: relative change on active region
        diff = psi[mask_active] - psi_old[mask_active]
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(psi_old[mask_active]), eps_norm))

        history["rel_change"].append(rel)
        history["psi_axis"].append(float(psi_axis))
        history["lin_info"].append(int(lin_info))

        if rel < tol:
            converged = True
            break

    # Final axis estimate with final psi
    psi_axis, axis_ij = _estimate_psi_axis(psi, mask_active, float(psi_lcfs))
    iz_ax, ir_ax = axis_ij
    axis_RZ = (float(R[ir_ax]), float(Z[iz_ax]))

    # Recompute jphi for returned state
    jphi = jphi_from_psi(
        psi=psi,
        RR=RR,
        psi_axis=psi_axis,
        psi_lcfs=float(psi_lcfs),
        profiles=profiles,
        clip_psin=clip_psin,
        mu0=MU0,
    )

    return {
        "psi": psi,
        "psi_axis": float(psi_axis),
        "axis_RZ": axis_RZ,
        "axis_ij": axis_ij,
        "mask": mask_active,
        "ring": ring,
        "jphi": jphi,
        "converged": bool(converged),
        "history": history,
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing gs_solve_fixed.py (smoke test)")

    # Minimal smoke test: solve with very mild profiles on a Miller-ish LCFS.
    # This checks that the code runs end-to-end (not that the equilibrium is physically tuned).

    from tokdesign.geometry.grid import make_rz_grid
    from tokdesign.geometry._plasma_boundary import miller_boundary
    from tokdesign.physics._gs_profiles import profiles_from_dict

    # Grid
    R, Z, RR, ZZ = make_rz_grid(0.8, 2.6, -1.2, 1.2, NR=80, NZ=81)

    # LCFS polyline
    lcfs = miller_boundary(R0=1.65, a=0.55, kappa=1.7, delta=0.33, npts=400)

    # Profiles (gentle)
    prof = profiles_from_dict({
        "pressure": {"model": "power", "p0": 2.0e4, "alpha_p": 1.5},
        "toroidal_field_function": {"model": "linear", "F0": 4.1, "alpha_F": 0.02},
    })

    cfg = {
        "max_picard_iter": 30,
        "tol_rel_change": 1e-6,
        "under_relaxation": 0.5,
        "linear_solver": "spsolve",
        "clip_psin": True,
    }

    out = solve_fixed_boundary_gs(
        R=R, Z=Z, RR=RR, ZZ=ZZ,
        lcfs_poly=lcfs,
        psi_lcfs=0.0,
        profiles=prof,
        solver_cfg=cfg,
    )

    print("  converged:", out["converged"])
    print("  final rel_change:", out["history"]["rel_change"][-1])
    print("  psi_axis:", out["psi_axis"])
    print("  axis_RZ:", out["axis_RZ"])
    assert out["psi"].shape == RR.shape
    assert out["jphi"].shape == RR.shape
    assert np.any(out["mask"])
    assert np.any(out["ring"])

    print("gs_solve_fixed.py smoke test passed.")
