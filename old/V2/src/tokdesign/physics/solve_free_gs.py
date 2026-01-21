"""
physics/solve_free_gs.py
========================

Free-boundary Grad–Shafranov (GS) equilibrium driver.

This module contains the *physics engine* for stage 05:
it iterates between vacuum flux from coils and a fixed-boundary GS solve
until the LCFS (last closed flux surface) is self-consistent.

Design philosophy (matches project docs):
- NO HDF5 I/O here (scripts handle that)
- NO plotting here
- NO optimization here
- PURE reusable physics logic

Expected to be called by:
  scripts/05_solve_free_gs.py  (thin orchestration)

Dependencies (expected in your repo):
- tokdesign.physics.gs_solve_fixed.solve_fixed_boundary_gs
- tokdesign.physics.lcfs.extract_lcfs
- tokdesign.physics.derived.compute_Ip  (optional, but used if available)

Self-test:
- Included at bottom; runs without the real GS solver by injecting a toy fixed-solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np


# =============================================================================
# Exceptions
# =============================================================================

class FreeBoundaryConvergenceError(RuntimeError):
    """Raised when the free-boundary outer iteration fails to converge."""
    pass


# =============================================================================
# Configuration helpers
# =============================================================================

@dataclass(frozen=True)
class FreeBoundaryConfig:
    """
    Configuration for the *outer* free-boundary iteration.

    Typical source: solver.yaml -> free_boundary
    """
    max_outer_iter: int = 50
    tol_lcfs_change: float = 1e-3         # meters (RMS displacement of LCFS)
    tol_Ip_rel_change: float = 1e-3       # relative change in Ip
    under_relax_lcfs: float = 0.5         # [0..1]
    under_relax_plasma: float = 0.7       # [0..1]
    lcfs_selector: str = "enclosing_axis"
    lcfs_resample_points: int = 128
    alpha_min: float = 0.05               # minimum relaxation if we start damping on failures


# =============================================================================
# Small numeric utilities
# =============================================================================

def _grid_spacing(R: np.ndarray, Z: np.ndarray) -> tuple[float, float]:
    """Return uniform grid spacing (dR, dZ)."""
    if R.size < 2 or Z.size < 2:
        raise ValueError("R and Z must have at least 2 points to compute spacing.")
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    return dR, dZ


def _psi_from_coils(G_psi: np.ndarray, I_pf: np.ndarray) -> np.ndarray:
    """
    Vacuum psi from coils using precomputed Green's function:
      G_psi: (Nc, NZ, NR)
      I_pf:  (Nc,)
    Returns psi_vac: (NZ, NR)
    """
    if G_psi.ndim != 3:
        raise ValueError(f"G_psi must be 3D (Nc,NZ,NR), got shape {G_psi.shape}")
    Nc = G_psi.shape[0]
    if I_pf.shape != (Nc,):
        raise ValueError(f"I_pf must have shape ({Nc},), got {I_pf.shape}")
    # Efficient contraction: sum_i G[i,:,:] * I[i]
    return np.tensordot(I_pf, G_psi, axes=(0, 0))


def _poly_rms_displacement(poly_new: np.ndarray, poly_old: np.ndarray) -> float:
    """
    RMS displacement between two polylines assumed to be pointwise comparable:
    shape (N,2) each, same N.
    """
    if poly_new.shape != poly_old.shape:
        raise ValueError("Polylines must have same shape for displacement metric.")
    d = poly_new - poly_old
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _rel_change(new: float, old: float, eps: float = 1e-12) -> float:
    """Relative change |new-old| / max(|old|, eps)."""
    return float(abs(new - old) / max(abs(old), eps))


def _blend_poly(alpha: float, new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Under-relaxation for polylines (same shape)."""
    if new.shape != old.shape:
        raise ValueError("Cannot blend polylines with different shapes.")
    return alpha * new + (1.0 - alpha) * old


# =============================================================================
# Main solver
# =============================================================================

def solve_free_boundary_equilibrium(
    R: np.ndarray,
    Z: np.ndarray,
    RR: np.ndarray,
    ZZ: np.ndarray,
    G_psi: np.ndarray,
    I_pf: np.ndarray,
    psi_lcfs: float,
    lcfs_init: np.ndarray,
    profile: Any,
    fixed_cfg: dict,
    free_cfg: FreeBoundaryConfig | dict | None = None,
    psi_plasma_init: np.ndarray | None = None,
    axis_init: tuple[float, float] | None = None,
    fixed_solver: Callable[..., dict] | None = None,
) -> dict:
    """
    Solve a free-boundary equilibrium by outer iteration.

    Parameters
    ----------
    R, Z : 1D arrays
        Grid coordinates (uniform spacing assumed).
    RR, ZZ : 2D arrays
        Meshgrid arrays with shape (NZ, NR).
    G_psi : ndarray
        Coil Green's function for psi per amp: shape (Nc, NZ, NR).
    I_pf : ndarray
        PF currents: shape (Nc,).
    psi_lcfs : float
        Flux value defining the LCFS (often 0).
    lcfs_init : ndarray
        Initial boundary polyline (N,2) [R,Z], e.g. from target boundary.
    profile : Any
        Profile parameters passed through to the fixed-boundary solver
        (could be a dataclass or dict, depending on your implementation).
    fixed_cfg : dict
        Configuration for the *fixed-boundary* GS solver (Picard, linear solver, etc).
    free_cfg : FreeBoundaryConfig or dict, optional
        Configuration for the *outer* free-boundary iteration.
    psi_plasma_init : ndarray, optional
        Initial guess for plasma psi (NZ, NR). If None, zeros are used.
    axis_init : (R,Z), optional
        Initial guess for magnetic axis. If None, uses centroid of lcfs_init.
    fixed_solver : callable, optional
        Dependency injection point; if None, imports the project's
        solve_fixed_boundary_gs at runtime.

    Returns
    -------
    result : dict
        {
          "equilibrium": {
              "psi_total", "psi_plasma", "psi_vac",
              "lcfs_poly", "psi_axis", "axis_RZ",
              "jphi", "mask",
          },
          "history": {
              "outer_iter", "lcfs_rms", "Ip", "dIp_rel",
              "alpha_lcfs", "alpha_plasma",
              "converged", "stop_reason",
          }
        }

    Notes
    -----
    - This function does NOT compute BR/BZ/Bphi. Stage 05 script can do that
      after convergence (physics/fields.py), then derived scalars (physics/derived.py).
    - LCFS extraction is delegated to tokdesign.physics.lcfs.extract_lcfs.
    """
    # -------------------------
    # Load/normalize free_cfg
    # -------------------------
    if free_cfg is None:
        free = FreeBoundaryConfig()
    elif isinstance(free_cfg, dict):
        free = FreeBoundaryConfig(**free_cfg)
    else:
        free = free_cfg

    # -------------------------
    # Import dependencies lazily (keeps module usable in selftest)
    # -------------------------
    if fixed_solver is None:
        from tokdesign.physics._gs_solve_fixed import solve_fixed_boundary_gs  # expected in your repo
        fixed_solver = solve_fixed_boundary_gs

    from tokdesign.physics._lcfs import extract_lcfs  # expected from your earlier implementation

    # Derived compute_Ip is optional; we fall back if not available.
    try:
        from tokdesign.physics._derived import compute_Ip
        have_compute_Ip = True
    except Exception:
        compute_Ip = None
        have_compute_Ip = False

    # -------------------------
    # Basic shape checks
    # -------------------------
    if RR.shape != ZZ.shape or RR.shape != (Z.size, R.size):
        raise ValueError("RR/ZZ must have shape (NZ,NR) matching Z and R.")
    if G_psi.shape[1:] != RR.shape:
        raise ValueError(f"G_psi spatial shape {G_psi.shape[1:]} must match RR {RR.shape}.")
    if lcfs_init.ndim != 2 or lcfs_init.shape[1] != 2:
        raise ValueError("lcfs_init must be (N,2) array of [R,Z] points.")

    # -------------------------
    # Initialize state
    # -------------------------
    dR, dZ = _grid_spacing(R, Z)

    psi_vac = _psi_from_coils(G_psi, I_pf)

    if psi_plasma_init is None:
        psi_plasma = np.zeros_like(psi_vac)
    else:
        if psi_plasma_init.shape != psi_vac.shape:
            raise ValueError("psi_plasma_init must have shape (NZ,NR).")
        psi_plasma = psi_plasma_init.copy()

    lcfs = lcfs_init.copy()

    if axis_init is None:
        # crude initial axis: centroid of initial boundary
        axis = (float(np.mean(lcfs[:, 0])), float(np.mean(lcfs[:, 1])))
    else:
        axis = axis_init

    # History buffers
    outer_iter_hist: list[int] = []
    lcfs_rms_hist: list[float] = []
    Ip_hist: list[float] = []
    dIp_rel_hist: list[float] = []
    alpha_lcfs_hist: list[float] = []
    alpha_plasma_hist: list[float] = []

    converged = False
    stop_reason = "max_outer_iter_reached"

    alpha_lcfs = float(free.under_relax_lcfs)
    alpha_plasma = float(free.under_relax_plasma)

    Ip_old = 0.0

    # -------------------------
    # Outer iteration loop
    # -------------------------
    for k in range(int(free.max_outer_iter)):
        # 1) total psi
        psi_total = psi_vac + psi_plasma
        if k==0: psi_total = psi_plasma

        # 2) Extract LCFS from psi_total
        lcfs_old = lcfs
        print("xxxxxxxxxxxxxxxx")

        psi_min = float(np.nanmin(psi_total))
        psi_max = float(np.nanmax(psi_total))
        print(f"[free-gs] k={k:03d} psi_lcfs={psi_lcfs:.6g} psi_min={psi_min:.6g} psi_max={psi_max:.6g} axis={axis}")
        try:
            lcfs_new = extract_lcfs(
                RR, ZZ, psi_total, psi_lcfs,
                axis_RZ=axis,
                selector=free.lcfs_selector,
                n_resample=free.lcfs_resample_points,
            )
        except Exception as e:
            # If LCFS extraction fails, damp relaxation and try again.
            print(f"[free-gs] k={k:03d} extract_lcfs FAILED: {type(e).__name__}: {e}")
            alpha_lcfs = max(free.alpha_min, 0.5 * alpha_lcfs)
            alpha_plasma = max(free.alpha_min, 0.5 * alpha_plasma)

            outer_iter_hist.append(k)
            lcfs_rms_hist.append(np.nan)
            Ip_hist.append(Ip_old)
            dIp_rel_hist.append(np.nan)
            alpha_lcfs_hist.append(alpha_lcfs)
            alpha_plasma_hist.append(alpha_plasma)

            # Continue; if we keep failing, outer loop ends and we error out.
            continue
        print("yyyyyyyyyyyyyyyyyy")
        # 3) Solve fixed-boundary GS inside lcfs_new
        fixed = fixed_solver(
            R, Z, RR, ZZ,
            lcfs_new,
            psi_lcfs,
            profile,
            fixed_cfg,
            psi_init=psi_plasma,
        )

        # Project's fixed solver is expected to return "psi", "axis_RZ", "psi_axis", "jphi", "mask"
        psi_plasma_new = fixed.get("psi", None)
        axis_new = fixed.get("axis_RZ", None)

        if psi_plasma_new is None or axis_new is None:
            raise RuntimeError(
                "Fixed-boundary solver did not return required keys "
                "('psi' and 'axis_RZ')."
            )

        # 4) Compute Ip (if possible). If not, we still converge on LCFS alone.
        Ip_new = Ip_old
        dIp_rel = np.nan
        if have_compute_Ip:
            jphi = fixed.get("jphi", None)
            mask = fixed.get("mask", None)
            if jphi is not None and mask is not None:
                Ip_new = float(compute_Ip(jphi, RR, dR, dZ, mask))
                dIp_rel = _rel_change(Ip_new, Ip_old)

        # 5) Under-relaxation
        psi_plasma = alpha_plasma * psi_plasma_new + (1.0 - alpha_plasma) * psi_plasma
        axis = (float(axis_new[0]), float(axis_new[1]))

        # 6) Convergence diagnostics
        # lcfs_old is the previous relaxed lcfs
        lcfs_relaxed = _blend_poly(alpha_lcfs, lcfs_new, lcfs_old)
        lcfs_rms = _poly_rms_displacement(lcfs_relaxed, lcfs_old)
        lcfs = lcfs_relaxed


        # Record history
        outer_iter_hist.append(k)
        lcfs_rms_hist.append(lcfs_rms)
        Ip_hist.append(Ip_new)
        dIp_rel_hist.append(dIp_rel)
        alpha_lcfs_hist.append(alpha_lcfs)
        alpha_plasma_hist.append(alpha_plasma)

        if (k % 5) == 0 or k == free.max_outer_iter - 1:
            print(f"[free-gs] k={k:03d} lcfs_rms={lcfs_rms:.3e} dIp_rel={dIp_rel:.3e} "
                f"alpha_lcfs={alpha_lcfs:.2f} alpha_plasma={alpha_plasma:.2f}")
            
        # Check convergence
        lcfs_ok = lcfs_rms < free.tol_lcfs_change
        Ip_ok = True
        if have_compute_Ip and np.isfinite(dIp_rel):
            Ip_ok = dIp_rel < free.tol_Ip_rel_change

        if lcfs_ok and Ip_ok:
            converged = True
            stop_reason = "converged"
            Ip_old = Ip_new
            break

        Ip_old = Ip_new

            
    if not converged:
        raise FreeBoundaryConvergenceError(
            f"Free-boundary solver did not converge after {free.max_outer_iter} iterations "
            f"(stop_reason={stop_reason})."
        )

    # Final recompute of psi_total using converged state
    psi_total = psi_vac + psi_plasma

    # One final LCFS extraction to ensure lcfs is consistent with final psi_total
    lcfs_final = extract_lcfs(
        RR, ZZ, psi_total, psi_lcfs,
        axis_RZ=axis,
        selector=free.lcfs_selector,
        n_resample=free.lcfs_resample_points,
    )

    # Package results
    equilibrium = {
        "psi_total": psi_total,
        "psi_plasma": psi_plasma,
        "psi_vac": psi_vac,
        "lcfs_poly": lcfs_final,
        "axis_RZ": axis,
        "psi_axis": fixed.get("psi_axis", np.nan),
        "jphi": fixed.get("jphi", None),
        "mask": fixed.get("mask", None),
    }

    history = {
        "outer_iter": np.asarray(outer_iter_hist, dtype=int),
        "lcfs_rms": np.asarray(lcfs_rms_hist, dtype=float),
        "Ip": np.asarray(Ip_hist, dtype=float),
        "dIp_rel": np.asarray(dIp_rel_hist, dtype=float),
        "alpha_lcfs": np.asarray(alpha_lcfs_hist, dtype=float),
        "alpha_plasma": np.asarray(alpha_plasma_hist, dtype=float),
        "converged": bool(converged),
        "stop_reason": str(stop_reason),
        "n_outer_iter": int(outer_iter_hist[-1] + 1 if outer_iter_hist else 0),
    }

    return {"equilibrium": equilibrium, "history": history}


# =============================================================================
# Self-test (runs without a real GS solver)
# =============================================================================

def _toy_fixed_solver(
    R: np.ndarray,
    Z: np.ndarray,
    RR: np.ndarray,
    ZZ: np.ndarray,
    lcfs_poly: np.ndarray,
    psi_lcfs: float,
    profile: Any,
    fixed_cfg: dict,
    psi_init: np.ndarray | None = None,
) -> dict:
    """
    A minimal stand-in for solve_fixed_boundary_gs for self-testing.

    It fabricates a "plasma" psi field whose LCFS is a circle estimated from the
    provided lcfs_poly. This is NOT physics; it just validates the free-boundary
    orchestration and LCFS coupling logic.

    Returned keys match the contract expected by solve_free_boundary_equilibrium.
    """
    # Estimate axis and minor radius from lcfs_poly
    R0 = float(np.mean(lcfs_poly[:, 0]))
    Z0 = float(np.mean(lcfs_poly[:, 1]))
    r = np.sqrt((lcfs_poly[:, 0] - R0) ** 2 + (lcfs_poly[:, 1] - Z0) ** 2)
    a = float(np.mean(r))

    # Construct psi_plasma such that psi_total contour at psi_lcfs is roughly that circle.
    # Here we ignore psi_vac; the outer solver will add psi_vac anyway.
    # We set psi_plasma = psi_lcfs - ((R-R0)^2 + (Z-Z0)^2 - a^2)
    # so that psi_plasma == psi_lcfs when (R-R0)^2 + (Z-Z0)^2 == a^2.
    psi_plasma = psi_lcfs - ((RR - R0) ** 2 + (ZZ - Z0) ** 2 - a ** 2)

    # Fake outputs
    return {
        "psi": psi_plasma,
        "axis_RZ": (R0, Z0),
        "psi_axis": float(np.max(psi_plasma)),
        "jphi": np.zeros_like(psi_plasma),
        "mask": np.ones_like(psi_plasma, dtype=bool),
        "converged": True,
        "history": {"note": "toy solver"},
    }


def _selftest_solve_free_boundary_equilibrium():
    """
    Self-test for solve_free_boundary_equilibrium.

    This is a *smoke test* that:
    - builds a grid
    - defines a trivial coil vacuum field (zero)
    - uses a circular target boundary as initial LCFS
    - injects a toy fixed-boundary solver that produces a consistent LCFS
    - checks that the outer loop converges and outputs have expected shapes
    """
    print("Running solve_free_gs self-test...")

    # Grid
    R = np.linspace(1.0, 2.4, 160)
    Z = np.linspace(-0.9, 0.9, 180)
    RR, ZZ = np.meshgrid(R, Z)

    # One coil, but set G_psi=0 so psi_vac=0
    Nc = 1
    G_psi = np.zeros((Nc, Z.size, R.size), dtype=float)
    I_pf = np.zeros((Nc,), dtype=float)

    # LCFS definition
    R0 = 1.7
    a = 0.5
    psi_lcfs = a * a

    # Initial boundary: circle
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    lcfs_init = np.column_stack([R0 + a * np.cos(theta), 0.0 + a * np.sin(theta)])

    # Minimal configs
    free_cfg = FreeBoundaryConfig(
        max_outer_iter=80,
        tol_lcfs_change=1e-2,
        tol_Ip_rel_change=1e-2,
        under_relax_lcfs=1.0,
        under_relax_plasma=1.0,
        lcfs_resample_points=128,
    )
    fixed_cfg = {}     # unused by toy solver
    profile = {}       # unused by toy solver

    # Solve
    out = solve_free_boundary_equilibrium(
        R, Z, RR, ZZ,
        G_psi, I_pf,
        psi_lcfs=psi_lcfs,
        lcfs_init=lcfs_init,
        profile=profile,
        fixed_cfg=fixed_cfg,
        free_cfg=free_cfg,
        fixed_solver=_toy_fixed_solver,   # key: inject toy solver
        axis_init=(R0, 0.0),
    )

    eq = out["equilibrium"]
    hist = out["history"]

    # Checks
    assert hist["converged"] is True, "Self-test: solver did not converge."
    assert eq["psi_total"].shape == (Z.size, R.size), "psi_total shape mismatch."
    assert eq["psi_plasma"].shape == (Z.size, R.size), "psi_plasma shape mismatch."
    assert eq["psi_vac"].shape == (Z.size, R.size), "psi_vac shape mismatch."
    assert eq["lcfs_poly"].shape == (free_cfg.lcfs_resample_points, 2), "lcfs_poly shape mismatch."

    # Boundary sanity: mean radius close to a
    lcfs = eq["lcfs_poly"]
    r = np.sqrt((lcfs[:, 0] - R0) ** 2 + lcfs[:, 1] ** 2)
    r_mean = float(np.mean(r))
    r_err = float(np.max(np.abs(r - a)))

    print(f"  converged in {hist['n_outer_iter']} outer iterations")
    print(f"  mean LCFS radius     : {r_mean:.6f} (target {a:.6f})")
    print(f"  max radius deviation : {r_err:.3e}")

    assert abs(r_mean - a) < 2e-2, "Self-test: LCFS mean radius off."
    assert r_err < 5e-2, "Self-test: LCFS deviation too large."

    print("solve_free_gs self-test PASSED.")


if __name__ == "__main__":
    _selftest_solve_free_boundary_equilibrium()
