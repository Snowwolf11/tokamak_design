"""
tokdesign.physics.equilibrium
=============================

Grad–Shafranov equilibrium utilities.

This module replaces the previous Stage-01 *stub* equilibrium generator with a
real (fixed-boundary) Grad–Shafranov solver on a uniform (R,Z) grid.

Scope (now)
-----------
• Fixed-boundary GS solve inside a prescribed LCFS boundary (a closed polyline).
• Picard iterations on the nonlinear source term j_phi(psi) provided by a profile model.
• Returns outputs compatible with Stage-01 optimization (`stage01_fixed.py`).

Scope (later / out of this file)
--------------------------------
• Free-boundary equilibrium (PF coils + vessel).
• More sophisticated solvers (Newton / multigrid / flux-surface averages).
• Rigorous computation of q, shear, alpha, beta, etc.

Conventions
-----------
We solve (axisymmetric, cylindrical):
    Δ* ψ = - μ0 R j_phi(ψ)

with:
    Δ* ψ = ∂²ψ/∂R² - (1/R) ∂ψ/∂R + ∂²ψ/∂Z²

Boundary condition:
    ψ = ψ_lcfs on the prescribed LCFS boundary.

Public API
----------
solve_fixed_equilibrium(params, grid, cfg_opt=None, psi_init=None) -> EquilibriumResult
make_equilibrium(params, grid, psi_init=None) -> dict   (lower-level; mostly for internal use)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from matplotlib.path import Path as MplPath

# =============================================================================
# Real imports from the modules we implemented
# =============================================================================

from tokdesign.physics.gs_profiles import GSProfiles, build_gs_profiles, normalize_psi
from tokdesign.physics.fields import poloidal_field_from_psi, toroidal_field_from_F
from tokdesign.physics.q import compute_q_profile
from tokdesign.physics.derived import compute_shear, compute_alpha
from tokdesign.physics.metrics import compute_equilibrium_scalars
from tokdesign.physics.boundary import build_lcfs_polyline_from_controls


MU0 = 4e-7 * np.pi


# =============================================================================
# Stage-01 result container expected by stage01_fixed.py
# =============================================================================

@dataclass
class EquilibriumResult:
    """
    Container returned by solve_fixed_equilibrium().

    This matches what `tokdesign.optimization.stage01_fixed` writes to HDF5.
    """
    # grid references (useful for debug; stage writes these from problem.grid anyway)
    RR: np.ndarray
    ZZ: np.ndarray

    # boundary and masks
    lcfs_R: np.ndarray
    lcfs_Z: np.ndarray
    plasma_mask: np.ndarray  # bool mask (stage writes as int8)

    # flux and current
    psi: np.ndarray
    psi_axis: float
    psi_lcfs: float
    j_phi: np.ndarray

    # axis location
    axis_R: float
    axis_Z: float

    # fields
    BR: np.ndarray
    BZ: np.ndarray
    Bphi: np.ndarray

    # profile-like fields on the grid
    psi_bar: np.ndarray
    rho: np.ndarray
    p: np.ndarray
    F: np.ndarray
    Te: np.ndarray
    n_e: np.ndarray

    q: np.ndarray
    s: np.ndarray
    alpha: np.ndarray

    # diagnostics
    gs_iterations: int
    residual_norm: float

    # metrics/scalars
    scalars: Dict[str, float]


# =============================================================================
# Public entry points
# =============================================================================

def solve_fixed_equilibrium(
    *,
    params: Dict[str, Any],
    grid: Any,
    cfg_opt: Optional[Dict[str, Any]] = None,
    psi_init: Optional[np.ndarray] = None,
) -> EquilibriumResult:
    """
    Stage-01 entry point.

    This is the function imported by `stage01_fixed.py`:
        from tokdesign.physics.equilibrium import solve_fixed_equilibrium

    Notes
    -----
    - cfg_opt is accepted for compatibility with the stage orchestrator; currently
      it is not needed here (the grid already contains discretization info).
    - No file writing occurs here. Stage 01 orchestrator owns write-out.
    """
    # Run the core equilibrium computation
    out = make_equilibrium(params=params, grid=grid, psi_init=psi_init)

    # Reconstruct what Stage 01 expects additionally: lcfs arrays + plasma mask + axis coords
    controls = params.get("controls", {}) if isinstance(params, dict) else {}
    if not isinstance(controls, dict):
        controls = {}

    boundary_cfg = controls.get("plasma_boundary", {}) if isinstance(controls.get("plasma_boundary", {}), dict) else {}
    lcfs_poly = boundary_cfg.get("polyline", None)
    if lcfs_poly is None:
        n = int(boundary_cfg.get("n", 256))
        lcfs_poly = build_lcfs_polyline_from_controls(controls, n=n)
    else:
        lcfs_poly = np.asarray(lcfs_poly, dtype=float)

    lcfs_poly = _ensure_closed_polyline(lcfs_poly)
    lcfs_R = np.asarray(lcfs_poly[:, 0], dtype=float)
    lcfs_Z = np.asarray(lcfs_poly[:, 1], dtype=float)

    RR = np.asarray(grid.RR, dtype=float)
    ZZ = np.asarray(grid.ZZ, dtype=float)
    plasma_mask = _mask_inside_polyline(RR, ZZ, lcfs_poly)

    # Axis location: use argmax(psi) inside mask (same convention as _estimate_psi_axis)
    psi = out["psi"]
    _, (iz_ax, ir_ax) = _estimate_psi_axis(psi, plasma_mask)
    axis_R = float(RR[iz_ax, ir_ax])
    axis_Z = float(ZZ[iz_ax, ir_ax])

    return EquilibriumResult(
        RR=RR,
        ZZ=ZZ,
        lcfs_R=lcfs_R,
        lcfs_Z=lcfs_Z,
        plasma_mask=plasma_mask,

        psi=out["psi"],
        psi_axis=float(_safe_float(out.get("scalars", {}).get("psi_axis", out.get("psi_axis", out["psi"][iz_ax, ir_ax])))),
        psi_lcfs=float(_safe_float(out.get("scalars", {}).get("psi_lcfs", out.get("psi_lcfs", controls.get("equilibrium", {}).get("psi_lcfs", 1.0))))),
        j_phi=out["j_phi"],

        axis_R=axis_R,
        axis_Z=axis_Z,

        BR=out["BR"],
        BZ=out["BZ"],
        Bphi=out["Bphi"],

        psi_bar=out["psi_bar"],
        rho=out["rho"],
        p=out["p"],
        F=out["F"],
        Te=out["Te"],
        n_e=out["n_e"],
        q=out["q"],
        s=out["s"],
        alpha=out["alpha"],

        gs_iterations=int(out["gs_iterations"]),
        residual_norm=float(out["residual_norm"]),

        scalars=out["scalars"],
    )


def make_equilibrium(
    params: Dict[str, Any],
    grid: Any,
    *,
    psi_init: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute a fixed-boundary Grad–Shafranov equilibrium on `grid`.

    Returns a dict of arrays and scalars. `solve_fixed_equilibrium()` wraps this
    dict into an EquilibriumResult with extra geometry fields.

    IMPORTANT (Stage-01 requirement)
    --------------------------------
    Stage01's `_finite_equilibrium(eq)` requires that *all returned arrays are finite*
    on the *full* grid (inside and outside the LCFS). Therefore we explicitly fill
    outside-plasma values with safe finite defaults.
    """
    controls = params.get("controls", {}) if isinstance(params, dict) else {}
    if not isinstance(controls, dict):
        controls = {}

    # ----------------------------
    # Boundary (LCFS)
    # ----------------------------
    boundary_cfg = (
        controls.get("plasma_boundary", {})
        if isinstance(controls.get("plasma_boundary", {}), dict)
        else {}
    )
    lcfs_poly = boundary_cfg.get("polyline", None)
    if lcfs_poly is None:
        n = int(boundary_cfg.get("n", 256))
        lcfs_poly = build_lcfs_polyline_from_controls(controls, n=n)
    else:
        lcfs_poly = np.asarray(lcfs_poly, dtype=float)

    # Ensure closed polyline (and stable inside-mask)
    lcfs_poly = _ensure_closed_polyline(np.asarray(lcfs_poly, dtype=float))

    RR = np.asarray(grid.RR, dtype=float)
    ZZ = np.asarray(grid.ZZ, dtype=float)
    plasma_mask = _mask_inside_polyline(RR, ZZ, lcfs_poly)
    outside = ~plasma_mask

    # Flux reference values (convention: psi_lcfs fixed; psi_axis found by solve)
    eq_cfg = (
        controls.get("equilibrium", {})
        if isinstance(controls.get("equilibrium", {}), dict)
        else {}
    )
    psi_lcfs = float(eq_cfg.get("psi_lcfs", 1.0))

    # ----------------------------
    # Profiles
    # ----------------------------
    # IMPORTANT: pass full controls so enforcement.I_t and B0 linkage work.
    profiles = build_gs_profiles(controls)

    # ----------------------------
    # Solver config
    # ----------------------------
    solver_cfg = controls.get("gs_solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {}
    cfg = GSSolverConfig(
        max_iter=int(solver_cfg.get("max_iter", 200)),
        tol_residual=float(solver_cfg.get("tol_residual", 1e-8)),
        tol_update=float(solver_cfg.get("tol_update", 1e-10)),
        under_relax=float(solver_cfg.get("under_relax", 0.5)),
        psi_axis_guess=solver_cfg.get("psi_axis_guess", None),
        dirichlet_ring=bool(solver_cfg.get("dirichlet_ring", True)),
    )

    # ----------------------------
    # Solve GS
    # ----------------------------
    sol = _solve_fixed_boundary_gs(
        R=np.asarray(grid.R, dtype=float),
        Z=np.asarray(grid.Z, dtype=float),
        RR=RR,
        ZZ=ZZ,
        lcfs_poly=lcfs_poly,
        psi_lcfs=psi_lcfs,
        profiles=profiles,
        cfg=cfg,
        psi_init=psi_init,
        dR=float(grid.dR),
        dZ=float(grid.dZ),
    )

    psi = sol.psi
    j_phi = sol.jphi
    psi_axis = sol.psi_axis

    # Force outside to boundary value (keeps everything finite and consistent)
    psi[outside] = psi_lcfs
    j_phi[outside] = 0.0

    # ----------------------------
    # Derived fields + profiles
    # ----------------------------
    BR, BZ = poloidal_field_from_psi(np.asarray(grid.R, float), np.asarray(grid.Z, float), psi)

    psi_bar = normalize_psi(psi, psi_axis, psi_lcfs, clip=True)
    rho = np.sqrt(np.clip(psi_bar, 0.0, 1.0))

    # p and F are allowed to be grid-shaped (Stage-01 uses rho grid)
    p = profiles.pressure(rho)
    F = profiles.toroidal_flux_function(rho)

    Te = profiles.temperature(rho)
    n_e = profiles.density(rho)


    # ------------------------------------------------------------------
    # Stage-01 finiteness contract: fill outside-plasma values
    # ------------------------------------------------------------------
    # Normalized flux and rho outside: treat as edge
    psi_bar[outside] = 1.0
    rho[outside] = 1.0

    # Pressure outside: vacuum
    p[outside] = 0.0

    Te[outside] = 0.0
    n_e[outside] = 0.0

    # Toroidal flux function outside: set to a safe edge/vacuum value.
    # Prefer an edge-band median (psi_bar>=0.95), else any finite median, else 0.
    edge_band = plasma_mask & (psi_bar >= 0.95)
    if np.any(edge_band) and np.any(np.isfinite(F[edge_band])):
        F_edge = float(np.nanmedian(F[edge_band]))
    elif np.any(np.isfinite(F[plasma_mask])):
        F_edge = float(np.nanmedian(F[plasma_mask]))
    elif np.any(np.isfinite(F)):
        F_edge = float(np.nanmedian(F[np.isfinite(F)]))
    else:
        F_edge = 0.0
    F[outside] = F_edge

    # Toroidal field (recompute after F fill to guarantee finiteness)
    Bphi = toroidal_field_from_F(RR, F)

    # q profile (may naturally return NaN outside; we will fill it)
    q = compute_q_profile(
        RR=RR,
        ZZ=ZZ,
        psi=psi,
        F=F,
        psi_axis=psi_axis,
        psi_lcfs=psi_lcfs,
        lcfs_poly=lcfs_poly,
    )

    # Fill q outside with a safe edge value (physically irrelevant outside plasma)
    if np.any(edge_band) and np.any(np.isfinite(q[edge_band])):
        q_edge = float(np.nanmedian(q[edge_band]))
    elif np.any(np.isfinite(q[plasma_mask])):
        q_edge = float(np.nanmedian(q[plasma_mask]))
    else:
        q_edge = 1.0
    q[outside] = q_edge

    # dp/drho:
    # Avoid `np.gradient(p, rho)` on 2D fields. Instead compute a robust 1D profile
    # p(rho) via binning, differentiate, and map back to the grid.
    dp_drho = _dp_drho_from_grid(p=p, rho=rho)

    # s, alpha (these may be 1D or 2D depending on implementation; we fill outside)
    s = compute_shear(q, rho)
    alpha = compute_alpha(rho=rho, q=q, dp_drho=dp_drho)

    # Ensure s/alpha are finite everywhere
    s = np.asarray(s, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if s.shape == psi.shape:
        s[outside] = 0.0
    if alpha.shape == psi.shape:
        alpha[outside] = 0.0

    # As a final safety net: replace any remaining NaN/inf with benign values
    # (This prevents a single numerical hiccup from killing the optimization run.)
    def _finite_fill(arr: np.ndarray, fill: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = fill
        return arr

    psi = _finite_fill(psi, psi_lcfs)
    j_phi = _finite_fill(j_phi, 0.0)
    BR = _finite_fill(BR, 0.0)
    BZ = _finite_fill(BZ, 0.0)
    Bphi = _finite_fill(Bphi, 0.0)
    psi_bar = _finite_fill(psi_bar, 1.0)
    rho = _finite_fill(rho, 1.0)
    p = _finite_fill(p, 0.0)
    F = _finite_fill(F, F_edge)
    q = _finite_fill(q, q_edge)
    n_e = _finite_fill(n_e, 0.0)
    Te = _finite_fill(Te, 0.0)
    if s.shape == psi.shape:
        s = _finite_fill(s, 0.0)
    if alpha.shape == psi.shape:
        alpha = _finite_fill(alpha, 0.0)

    # ----------------------------
    # Scalars (metrics / optimization terms)
    # ----------------------------
    scalars = compute_equilibrium_scalars(
        controls=controls,
        grid=grid,
        lcfs_poly=lcfs_poly,
        psi=psi,
        j_phi=j_phi,
        BR=BR,
        BZ=BZ,
        Bphi=Bphi,
        psi_axis=float(psi_axis),
        psi_lcfs=float(psi_lcfs),
        psi_bar=psi_bar,
        rho=rho,
        p=p,
        F=F,
        Te=Te.astype(float),
        n_e=n_e.astype(float),
        q=q,
        s=s,
        alpha=alpha,
        gs_iterations=int(sol.iterations),
        residual_norm=float(sol.residual_norm),
    )

    # Helpful (and harmless): store these in scalars too for stage traces
    scalars.setdefault("psi_axis", float(psi_axis))
    scalars.setdefault("psi_lcfs", float(psi_lcfs))

    return dict(
        psi=psi.astype(float),
        j_phi=j_phi.astype(float),
        BR=BR.astype(float),
        BZ=BZ.astype(float),
        Bphi=Bphi.astype(float),
        psi_bar=psi_bar.astype(float),
        rho=rho.astype(float),
        p=p.astype(float),
        F=F.astype(float),
        Te=Te.astype(float),
        n_e=n_e.astype(float),
        q=q.astype(float),
        s=np.asarray(s, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        gs_iterations=int(sol.iterations),
        residual_norm=float(sol.residual_norm),
        scalars=scalars,
    )


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass(frozen=True)
class GSSolverConfig:
    """Numerics for Picard fixed-boundary GS solve."""
    max_iter: int = 200
    tol_residual: float = 1e-8
    tol_update: float = 1e-10
    under_relax: float = 0.5  # 0<ω<=1
    psi_axis_guess: Optional[float] = None
    dirichlet_ring: bool = True


# =============================================================================
# Fixed-boundary GS solver (Picard)
# =============================================================================

@dataclass
class _GSSolution:
    psi: np.ndarray  # (NZ,NR)
    jphi: np.ndarray  # (NZ,NR)
    psi_axis: float
    axis_iz_ir: Tuple[int, int]
    iterations: int
    residual_norm: float


def _solve_fixed_boundary_gs(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    RR: np.ndarray,
    ZZ: np.ndarray,
    lcfs_poly: np.ndarray,
    psi_lcfs: float,
    profiles: GSProfiles,
    cfg: GSSolverConfig,
    psi_init: Optional[np.ndarray],
    dR: float,
    dZ: float,
) -> _GSSolution:
    """
    Picard (fixed-point) solve of the fixed-boundary GS equation.

    Implementation details:
    • Build an "inside LCFS" mask.
    • Identify a Dirichlet ring on the inside of the boundary (optional).
    • Assemble sparse Δ* operator once for unknown points.
    • Iterate:
        jphi <- profiles.jphi(psi)
        solve A psi = b(jphi) with Dirichlet contributions
        under-relaxation
    """
    lcfs_poly = _ensure_closed_polyline(np.asarray(lcfs_poly, dtype=float))
    mask_inside = _mask_inside_polyline(RR, ZZ, lcfs_poly)

    if cfg.dirichlet_ring:
        mask_dir = _dirichlet_ring_from_mask(mask_inside)
    else:
        mask_dir = np.zeros_like(mask_inside, dtype=bool)

    mask_active = mask_inside & (~mask_dir)
    if not np.any(mask_active):
        raise ValueError("No active points inside LCFS after applying Dirichlet ring.")

    fwd, rev = _build_active_maps(mask_active)

    A, b_dir = _assemble_delta_star_operator(
        R=R,
        mask_active=mask_active,
        mask_dir=mask_dir,
        fwd_map=fwd,
        rev_map=rev,
        dR=dR,
        dZ=dZ,
        psi_dirichlet=psi_lcfs,
    )

    psi = np.full_like(RR, fill_value=float(psi_lcfs), dtype=float)
    if psi_init is not None:
        psi_init = np.asarray(psi_init, dtype=float)
        if psi_init.shape != psi.shape:
            raise ValueError(f"psi_init has shape {psi_init.shape}, expected {psi.shape}.")
        psi[mask_inside] = psi_init[mask_inside]

    if cfg.psi_axis_guess is not None:
        psi_axis0 = float(cfg.psi_axis_guess)
        psi[mask_inside] = psi_axis0 + (psi_lcfs - psi_axis0) * 0.8

    residual_norm = np.inf
    for k in range(cfg.max_iter):
        psi_old = psi.copy()

        psi_axis, (iz_ax, ir_ax) = _estimate_psi_axis(psi, mask_inside)

        jphi = np.zeros_like(psi)
        # NOTE: equilibrium calls profiles.jphi on (possibly) 1D masked arrays.
        # Our GSProfiles implementation supports that.
        jphi[mask_inside] = profiles.jphi(
            RR[mask_inside],
            ZZ[mask_inside],
            psi[mask_inside],
            psi_axis,
            psi_lcfs,
        )

        b_src_full = np.zeros_like(psi)
        b_src_full[mask_active] = -MU0 * RR[mask_active] * jphi[mask_active]
        b = b_dir + _gather_active(b_src_full, rev)

        psi_active_new = spsolve(A, b)

        psi_new = psi.copy()
        _scatter_active(psi_new, psi_active_new, rev)
        psi_new[mask_dir] = psi_lcfs
        psi_new[~mask_inside] = psi_lcfs

        omega = float(np.clip(cfg.under_relax, 1e-3, 1.0))
        psi = (1.0 - omega) * psi_old + omega * psi_new

        r = A @ _gather_active(psi, rev) - b
        denom = float(np.linalg.norm(b) + 1e-30)
        residual_norm = float(np.linalg.norm(r) / denom)

        du = float(
            np.linalg.norm(_gather_active(psi - psi_old, rev))
            / (np.linalg.norm(_gather_active(psi, rev)) + 1e-30)
        )

        if residual_norm < cfg.tol_residual and du < cfg.tol_update:
            return _GSSolution(
                psi=psi,
                jphi=jphi,
                psi_axis=float(psi_axis),
                axis_iz_ir=(int(iz_ax), int(ir_ax)),
                iterations=k + 1,
                residual_norm=float(residual_norm),
            )

    psi_axis, (iz_ax, ir_ax) = _estimate_psi_axis(psi, mask_inside)
    jphi = np.zeros_like(psi)
    jphi[mask_inside] = profiles.jphi(
        RR[mask_inside],
        ZZ[mask_inside],
        psi[mask_inside],
        psi_axis,
        psi_lcfs,
    )
    return _GSSolution(
        psi=psi,
        jphi=jphi,
        psi_axis=float(psi_axis),
        axis_iz_ir=(int(iz_ax), int(ir_ax)),
        iterations=cfg.max_iter,
        residual_norm=float(residual_norm),
    )


# =============================================================================
# Discretization helpers
# =============================================================================

def _ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("lcfs_poly must have shape (N,2) with columns [R,Z].")
    if poly.shape[0] < 4:
        raise ValueError("lcfs_poly must have at least 4 points.")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _mask_inside_polyline(RR: np.ndarray, ZZ: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return boolean mask of points inside the closed polyline."""
    path = MplPath(poly)
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    inside = path.contains_points(pts)
    return inside.reshape(RR.shape)


def _dirichlet_ring_from_mask(mask_inside: np.ndarray) -> np.ndarray:
    """
    Compute a "one-cell thick" ring on the inside of the LCFS.
    A point is in the ring if it is inside, but has any 4-neighbor outside.
    """
    inside = mask_inside
    ring = np.zeros_like(inside, dtype=bool)
    nz, nr = inside.shape
    for iz in range(1, nz - 1):
        for ir in range(1, nr - 1):
            if not inside[iz, ir]:
                continue
            if (
                (not inside[iz, ir - 1])
                or (not inside[iz, ir + 1])
                or (not inside[iz - 1, ir])
                or (not inside[iz + 1, ir])
            ):
                ring[iz, ir] = True
    return ring


def _build_active_maps(mask_active: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward map fwd[iz,ir] -> k (or -1). Reverse map rev[k] -> (iz,ir)."""
    nz, nr = mask_active.shape
    fwd = -np.ones((nz, nr), dtype=int)
    idxs = np.argwhere(mask_active)
    rev = np.zeros((idxs.shape[0], 2), dtype=int)
    for k, (iz, ir) in enumerate(idxs):
        fwd[iz, ir] = k
        rev[k, 0] = iz
        rev[k, 1] = ir
    return fwd, rev


def _assemble_delta_star_operator(
    *,
    R: np.ndarray,
    mask_active: np.ndarray,
    mask_dir: np.ndarray,
    fwd_map: np.ndarray,
    rev_map: np.ndarray,
    dR: float,
    dZ: float,
    psi_dirichlet: float,
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Assemble sparse matrix A approximating Δ* on active points.

    Neighbors that are Dirichlet (mask_dir or outside LCFS) contribute to RHS.
    Returns:
        A (csr), b_dir (N_active,)
    """
    nz, nr = mask_active.shape
    n = int(rev_map.shape[0])
    A = lil_matrix((n, n), dtype=float)
    b_dir = np.zeros((n,), dtype=float)

    inv_dR2 = 1.0 / (dR * dR)
    inv_dZ2 = 1.0 / (dZ * dZ)

    for k in range(n):
        iz, ir = int(rev_map[k, 0]), int(rev_map[k, 1])
        Ri = float(R[ir])

        # Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
        cRp = inv_dR2 - 1.0 / (2.0 * Ri * dR)
        cRm = inv_dR2 + 1.0 / (2.0 * Ri * dR)
        cZp = inv_dZ2
        cZm = inv_dZ2
        c0 = -2.0 * (inv_dR2 + inv_dZ2)

        A[k, k] = c0

        _accum_neighbor(A, b_dir, k, iz, ir - 1, cRm, fwd_map, mask_active, mask_dir, psi_dirichlet)
        _accum_neighbor(A, b_dir, k, iz, ir + 1, cRp, fwd_map, mask_active, mask_dir, psi_dirichlet)
        _accum_neighbor(A, b_dir, k, iz - 1, ir, cZm, fwd_map, mask_active, mask_dir, psi_dirichlet)
        _accum_neighbor(A, b_dir, k, iz + 1, ir, cZp, fwd_map, mask_active, mask_dir, psi_dirichlet)

    return A.tocsr(), b_dir


def _accum_neighbor(
    A: lil_matrix,
    b_dir: np.ndarray,
    k_center: int,
    iz_n: int,
    ir_n: int,
    coeff: float,
    fwd_map: np.ndarray,
    mask_active: np.ndarray,
    mask_dir: np.ndarray,
    psi_dirichlet: float,
) -> None:
    """Add neighbor contribution into A (if active) or RHS (if Dirichlet)."""
    nz, nr = mask_active.shape
    if iz_n < 0 or iz_n >= nz or ir_n < 0 or ir_n >= nr:
        b_dir[k_center] -= coeff * psi_dirichlet
        return

    if mask_active[iz_n, ir_n]:
        k_n = int(fwd_map[iz_n, ir_n])
        A[k_center, k_n] = coeff
    else:
        b_dir[k_center] -= coeff * psi_dirichlet


def _gather_active(field_full: np.ndarray, rev_map: np.ndarray) -> np.ndarray:
    """Gather full-grid values into active vector."""
    out = np.empty((rev_map.shape[0],), dtype=float)
    for k in range(rev_map.shape[0]):
        iz, ir = int(rev_map[k, 0]), int(rev_map[k, 1])
        out[k] = float(field_full[iz, ir])
    return out


def _scatter_active(field_full: np.ndarray, vec_active: np.ndarray, rev_map: np.ndarray) -> None:
    """Scatter active vector into full-grid array in-place."""
    for k in range(rev_map.shape[0]):
        iz, ir = int(rev_map[k, 0]), int(rev_map[k, 1])
        field_full[iz, ir] = float(vec_active[k])


def _estimate_psi_axis(psi: np.ndarray, mask_inside: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    """
    Estimate magnetic axis as an extremum of psi inside the LCFS.

    By default we choose the *maximum* of ψ inside the plasma.
    If your convention uses the minimum as axis, switch argmax -> argmin and
    -inf -> +inf.
    """
    if psi.shape != mask_inside.shape:
        raise ValueError(f"psi shape {psi.shape} does not match mask shape {mask_inside.shape}")

    if not np.any(mask_inside):
        flat = int(np.argmax(psi))
        iz, ir = np.unravel_index(flat, psi.shape)
        return float(psi[iz, ir]), (int(iz), int(ir))

    masked = np.where(mask_inside, psi, -np.inf)
    flat = int(np.argmax(masked))
    iz, ir = np.unravel_index(flat, psi.shape)
    return float(psi[iz, ir]), (int(iz), int(ir))


# =============================================================================
# Small utilities
# =============================================================================

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float(np.nan)


def _dp_drho_from_grid(
    *,
    p: np.ndarray,
    rho: np.ndarray,
    nbins: int = 80,
) -> np.ndarray:
    """
    Robust dp/drho for grid-shaped p and rho.

    Strategy:
      1) Compute binned mean p(rho) on uniform rho bins in [0,1]
      2) Differentiate p(rho) w.r.t rho
      3) Map dp/drho back to the grid by interpolation in rho

    This avoids ambiguous `np.gradient(p, rho)` on 2D fields.
    """
    p = np.asarray(p, float)
    rho = np.asarray(rho, float)
    if p.shape != rho.shape:
        raise ValueError("p and rho must have the same shape for dp/drho mapping.")

    rr = np.clip(rho.ravel(), 0.0, 1.0)
    pp = p.ravel()

    edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full((nbins,), np.nan, dtype=float)

    idx = np.digitize(rr, edges) - 1
    ok = (idx >= 0) & (idx < nbins) & np.isfinite(rr) & np.isfinite(pp)
    for k in range(nbins):
        sel = ok & (idx == k)
        if np.any(sel):
            means[k] = float(np.mean(pp[sel]))

    valid = np.isfinite(means)
    if np.sum(valid) >= 2:
        means = np.interp(centers, centers[valid], means[valid])
    elif np.sum(valid) == 1:
        means[:] = means[valid][0]
    else:
        # can't compute; return zeros (safe fallback)
        return np.zeros_like(p)

    dp = np.gradient(means, centers + 1e-30, edge_order=1)

    dp_grid = np.empty_like(p, dtype=float)
    dp_grid.flat[:] = np.interp(rr, centers, dp).astype(float)
    return dp_grid
