"""
gs_operator.py
==============

Finite-difference operator utilities for the axisymmetric Grad–Shafranov equation.

We discretize the Grad–Shafranov operator:

    Δ*ψ = R ∂/∂R ( (1/R) ∂ψ/∂R ) + ∂²ψ/∂Z²

on a *uniform* rectangular R–Z grid with conventions from geometry/grids.py:

- R: shape (NR,)
- Z: shape (NZ,)
- 2D fields (psi, mask, etc.) are shape (NZ, NR) with indexing [iz, ir]

Key design choice (important for fixed-boundary solves)
-------------------------------------------------------
We build a sparse matrix **only for active/unknown nodes** (mask == True).
Neighbors that are inactive (mask == False) are *not* included in the matrix.
Their contribution is handled by apply_dirichlet(...), which shifts terms to RHS.

Boundary nodes (grid edges)
---------------------------
If a node is active but lies on the outer grid boundary (ir==0, ir==NR-1,
iz==0, iz==NZ-1), we place an identity row (psi = rhs), i.e. A[k,k] = 1.
This makes it easy to enforce Dirichlet values by setting rhs accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


Index2 = Tuple[int, int]  # (iz, ir)


def build_index_map(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a mapping between 2D grid indices and 1D unknown-vector indices.

    Parameters
    ----------
    mask : np.ndarray, shape (NZ, NR), dtype=bool
        True for active/unknown nodes, False for inactive/fixed nodes.

    Returns
    -------
    idx_map : np.ndarray, shape (NZ, NR), dtype=int
        idx_map[iz, ir] = k for active nodes, -1 for inactive nodes.
    rev_map : np.ndarray, shape (N_active, 2), dtype=int
        rev_map[k] = [iz, ir]
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D boolean array of shape (NZ, NR).")

    NZ, NR = mask.shape
    idx_map = -np.ones((NZ, NR), dtype=int)

    active = np.argwhere(mask)  # (N_active, 2) with rows [iz, ir]
    for k, (iz, ir) in enumerate(active):
        idx_map[iz, ir] = k

    rev_map = active.astype(int)
    return idx_map, rev_map


def build_delta_star_matrix(
    R: np.ndarray,
    Z: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> sp.csr_matrix:
    """
    Build sparse matrix A for the Δ* operator on a uniform grid.

    Unknown vector ordering is defined by build_index_map(mask), i.e.
    row-major in the list returned by np.argwhere(mask).

    Parameters
    ----------
    R : np.ndarray, shape (NR,)
        1D radial coordinate array [m], strictly increasing, uniform.
    Z : np.ndarray, shape (NZ,)
        1D vertical coordinate array [m], strictly increasing, uniform.
    mask : Optional[np.ndarray], shape (NZ, NR), dtype=bool
        Active/unknown nodes. If None, all nodes are active.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (N_active, N_active)
        Sparse operator matrix. For active nodes on the *outer grid boundary*,
        an identity row is used (Dirichlet-friendly).

    Notes
    -----
    Discretization (second order):
      Let i index R (ir), j index Z (iz). R_i = R[i].
      Define half-step radii:
        R_{i+1/2} = (R_i + R_{i+1})/2
        R_{i-1/2} = (R_{i-1} + R_i)/2

      Then:
        (Δ*_R ψ)_i ≈ (1/dR^2) * [ (R_{i+1/2}/R_i) ψ_{i+1}
                                -((R_{i+1/2}+R_{i-1/2})/R_i) ψ_i
                                + (R_{i-1/2}/R_i) ψ_{i-1} ]

        (Δ*_Z ψ)_j ≈ (ψ_{j+1} - 2 ψ_j + ψ_{j-1}) / dZ^2
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if R.ndim != 1 or Z.ndim != 1:
        raise ValueError("R and Z must be 1D arrays.")
    if R.size < 2 or Z.size < 2:
        raise ValueError("R and Z must have at least 2 points.")
    if not np.all(np.diff(R) > 0):
        raise ValueError("R must be strictly increasing.")
    if not np.all(np.diff(Z) > 0):
        raise ValueError("Z must be strictly increasing.")

    dR_arr = np.diff(R)
    dZ_arr = np.diff(Z)
    dR = float(np.mean(dR_arr))
    dZ = float(np.mean(dZ_arr))
    if not np.allclose(dR_arr, dR, rtol=1e-10, atol=0.0):
        raise ValueError("R grid is not uniform.")
    if not np.allclose(dZ_arr, dZ, rtol=1e-10, atol=0.0):
        raise ValueError("Z grid is not uniform.")

    NR = R.size
    NZ = Z.size

    if mask is None:
        mask = np.ones((NZ, NR), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (NZ, NR):
            raise ValueError(f"mask must have shape (NZ, NR)=({NZ},{NR}), got {mask.shape}")

    idx_map, rev_map = build_index_map(mask)
    n = rev_map.shape[0]

    # Assemble with COO triplets
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    inv_dR2 = 1.0 / (dR * dR)
    inv_dZ2 = 1.0 / (dZ * dZ)

    for k in range(n):
        iz, ir = int(rev_map[k, 0]), int(rev_map[k, 1])

        # Outer grid boundary: identity row (Dirichlet-friendly)
        if ir == 0 or ir == NR - 1 or iz == 0 or iz == NZ - 1:
            rows.append(k)
            cols.append(k)
            data.append(1.0)
            continue

        Ri = float(R[ir])
        if Ri <= 0.0:
            raise ValueError(
                f"Non-positive R encountered at ir={ir} (R={Ri}). "
                "Tokamak GS typically assumes R>0."
            )

        # Half-step radii
        Rph = 0.5 * (R[ir] + R[ir + 1])
        Rmh = 0.5 * (R[ir - 1] + R[ir])

        # Stencil coefficients
        cE = (Rph / Ri) * inv_dR2
        cW = (Rmh / Ri) * inv_dR2
        cN = inv_dZ2
        cS = inv_dZ2
        c0 = -(cE + cW) - 2.0 * inv_dZ2

        # Center
        rows.append(k)
        cols.append(k)
        data.append(c0)

        # East (ir+1)
        ke = idx_map[iz, ir + 1]
        if ke >= 0:
            rows.append(k)
            cols.append(int(ke))
            data.append(cE)

        # West (ir-1)
        kw = idx_map[iz, ir - 1]
        if kw >= 0:
            rows.append(k)
            cols.append(int(kw))
            data.append(cW)

        # North (iz+1)
        kn = idx_map[iz + 1, ir]
        if kn >= 0:
            rows.append(k)
            cols.append(int(kn))
            data.append(cN)

        # South (iz-1)
        ks = idx_map[iz - 1, ir]
        if ks >= 0:
            rows.append(k)
            cols.append(int(ks))
            data.append(cS)

    A = sp.coo_matrix((np.asarray(data, float), (np.asarray(rows, int), np.asarray(cols, int))), shape=(n, n))
    return A.tocsr()


def apply_dirichlet(
    rhs: np.ndarray,
    idx_map: np.ndarray,
    nodes: Sequence[Index2],
    values: Sequence[float],
    *,
    R: Optional[np.ndarray] = None,
    Z: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply Dirichlet boundary values by shifting neighbor contributions to RHS.

    This assumes:
    - You built A only over active nodes (mask True).
    - Inactive nodes (mask False) were NOT included in A.
    - Some inactive nodes adjacent to active nodes have known (Dirichlet) ψ values.

    Parameters
    ----------
    rhs : np.ndarray, shape (N_active,)
        RHS vector to be modified in-place (a copy is returned).
    idx_map : np.ndarray, shape (NZ, NR), dtype=int
        Mapping from (iz, ir) to unknown index k (>=0), -1 for inactive.
    nodes : Sequence[(iz, ir)]
        Grid nodes where ψ is prescribed.
    values : Sequence[float]
        Prescribed ψ values corresponding to nodes.
    R, Z : Optional[np.ndarray]
        If provided, we recompute the same stencil coefficients used in
        build_delta_star_matrix() to compute the RHS shift robustly.
        If not provided, we assume uniform grid and infer dR,dZ from idx_map
        is impossible, so we require R and Z for correctness.

    Returns
    -------
    rhs_mod : np.ndarray, shape (N_active,)
        Modified RHS.

    Notes
    -----
    For each active node k, if a neighbor is a Dirichlet node with value ψ_b,
    the term A[k,neighbor]*ψ_b is moved to RHS:

        rhs[k] -= coeff * ψ_b

    We only consider the 4-point (E,W,N,S) neighbors consistent with the operator.
    """
    rhs = np.asarray(rhs, dtype=float).copy()
    if rhs.ndim != 1:
        raise ValueError("rhs must be a 1D array of length N_active.")
    if idx_map.ndim != 2:
        raise ValueError("idx_map must be a 2D array of shape (NZ, NR).")
    if len(nodes) != len(values):
        raise ValueError("nodes and values must have the same length.")
    if R is None or Z is None:
        raise ValueError("apply_dirichlet requires R and Z to compute stencil coefficients.")

    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    NR = R.size
    NZ = Z.size
    if idx_map.shape != (NZ, NR):
        raise ValueError(f"idx_map shape {idx_map.shape} does not match (NZ,NR)=({NZ},{NR}).")

    dR_arr = np.diff(R)
    dZ_arr = np.diff(Z)
    dR = float(np.mean(dR_arr))
    dZ = float(np.mean(dZ_arr))
    if not np.allclose(dR_arr, dR, rtol=1e-10, atol=0.0):
        raise ValueError("R grid is not uniform.")
    if not np.allclose(dZ_arr, dZ, rtol=1e-10, atol=0.0):
        raise ValueError("Z grid is not uniform.")

    inv_dR2 = 1.0 / (dR * dR)
    inv_dZ2 = 1.0 / (dZ * dZ)

    # Map dirichlet nodes to values for fast lookup
    dir_map: Dict[Index2, float] = {}
    for (iz, ir), v in zip(nodes, values):
        dir_map[(int(iz), int(ir))] = float(v)

    # For each Dirichlet node, adjust RHS for each adjacent active node.
    for (iz_b, ir_b), psi_b in dir_map.items():
        # Skip anything out-of-bounds silently
        if not (0 <= iz_b < NZ and 0 <= ir_b < NR):
            continue

        # Adjacent active nodes (k) that reference this boundary node as neighbor:
        adjacent: List[Index2] = [
            (iz_b, ir_b - 1),  # boundary is east neighbor of this node
            (iz_b, ir_b + 1),  # boundary is west neighbor
            (iz_b - 1, ir_b),  # boundary is north neighbor
            (iz_b + 1, ir_b),  # boundary is south neighbor
        ]

        for iz, ir in adjacent:
            if not (0 <= iz < NZ and 0 <= ir < NR):
                continue
            k = idx_map[iz, ir]
            if k < 0:
                continue  # not an active unknown

            # If the adjacent active node is itself on outer boundary and you used identity rows,
            # we leave it to the caller to set rhs directly for that node.
            if ir == 0 or ir == NR - 1 or iz == 0 or iz == NZ - 1:
                continue

            Ri = float(R[ir])
            Rph = 0.5 * (R[ir] + R[ir + 1])
            Rmh = 0.5 * (R[ir - 1] + R[ir])

            cE = (Rph / Ri) * inv_dR2
            cW = (Rmh / Ri) * inv_dR2
            cN = inv_dZ2
            cS = inv_dZ2

            # Determine which neighbor the boundary node is, relative to (iz,ir)
            coeff = 0.0
            if (iz_b, ir_b) == (iz, ir + 1):  # east neighbor
                coeff = cE
            elif (iz_b, ir_b) == (iz, ir - 1):  # west neighbor
                coeff = cW
            elif (iz_b, ir_b) == (iz + 1, ir):  # north neighbor
                coeff = cN
            elif (iz_b, ir_b) == (iz - 1, ir):  # south neighbor
                coeff = cS
            else:
                continue

            rhs[int(k)] -= coeff * psi_b

    return rhs


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Testing gs_operator.py")

    import scipy.sparse.linalg as spla

    # -----------------------------
    # Build a small uniform grid
    # -----------------------------
    R = np.linspace(1.0, 2.0, 12)     # NR=12 (R>0)
    Z = np.linspace(-0.5, 0.5, 11)    # NZ=11
    NR = R.size
    NZ = Z.size

    RR, ZZ = np.meshgrid(R, Z, indexing="xy")  # shapes (NZ, NR)

    # -----------------------------
    # Define an "active" interior region:
    # - keep a 2-cell margin inactive so:
    #   (a) active nodes aren't on outer grid boundary
    #   (b) we can create an inactive Dirichlet "ring" around the active region
    # -----------------------------
    mask = np.zeros((NZ, NR), dtype=bool)
    mask[2:-2, 2:-2] = True  # active unknowns are the central block

    # Build maps and operator
    idx_map, rev_map = build_index_map(mask)
    A = build_delta_star_matrix(R, Z, mask=mask)

    n_active = rev_map.shape[0]
    assert A.shape == (n_active, n_active)
    assert np.all((idx_map[mask] >= 0))
    assert np.all((idx_map[~mask] == -1))

    # Check map invertibility consistency
    for k in [0, n_active // 3, 2 * n_active // 3, n_active - 1]:
        iz, ir = map(int, rev_map[k])
        assert idx_map[iz, ir] == k

    print(f"  Grid (NZ,NR)=({NZ},{NR}), active unknowns: {n_active}")
    print(f"  Operator shape: {A.shape}, nnz={A.nnz}")

    # -----------------------------
    # Manufactured solution test:
    # psi_exact(R,Z) = Z^2  =>  Δ*psi = 2 everywhere (for this operator)
    # -----------------------------
    psi_exact = ZZ**2

    # Build RHS vector for active nodes: rhs_k = 2
    rhs = np.full(n_active, 2.0, dtype=float)

    # -----------------------------
    # Create Dirichlet boundary nodes:
    # We take ALL inactive nodes that are immediate 4-neighbors of any active node.
    # Those are the "ring" that couples into the stencil.
    # -----------------------------
    ring_nodes = []
    ring_vals = []
    for iz in range(NZ):
        for ir in range(NR):
            if mask[iz, ir]:
                continue  # active nodes are unknowns, not Dirichlet here

            # Is this inactive node adjacent to an active node?
            adj = False
            for dz, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                iz2 = iz + dz
                ir2 = ir + dr
                if 0 <= iz2 < NZ and 0 <= ir2 < NR and mask[iz2, ir2]:
                    adj = True
                    break
            if adj:
                ring_nodes.append((iz, ir))
                ring_vals.append(float(psi_exact[iz, ir]))

    assert len(ring_nodes) > 0, "Did not find a Dirichlet ring; mask construction is wrong."

    # Apply Dirichlet shift
    rhs_mod = apply_dirichlet(rhs, idx_map, ring_nodes, ring_vals, R=R, Z=Z)

    # This MUST be non-zero, otherwise apply_dirichlet didn't do anything meaningful.
    shift_norm = np.linalg.norm(rhs_mod - rhs)
    print(f"  Dirichlet ring nodes: {len(ring_nodes)}")
    print(f"  RHS shift norm: {shift_norm:.6e}")
    assert shift_norm > 0.0

    # -----------------------------
    # Solve the linear system for the active unknowns
    # -----------------------------
    x = spla.spsolve(A, rhs_mod)

    # Reconstruct full psi field, fill active nodes with solution and inactive with exact
    psi_num = psi_exact.copy()
    for k in range(n_active):
        iz, ir = map(int, rev_map[k])
        psi_num[iz, ir] = x[k]

    # -----------------------------
    # Check error on the active region
    # -----------------------------
    err = psi_num[mask] - psi_exact[mask]
    err_l2 = float(np.sqrt(np.mean(err**2)))
    err_linf = float(np.max(np.abs(err)))

    print(f"  Error on active nodes: L2={err_l2:.6e}, Linf={err_linf:.6e}")

    # Tolerances: depends on grid, but should be small for a smooth quadratic solution.
    # This is finite-difference + masked Dirichlet, so we expect good agreement.
    assert err_linf < 1e-3, "Operator/Dirichlet test failed: error too large."

    print("gs_operator.py self-test passed.")
