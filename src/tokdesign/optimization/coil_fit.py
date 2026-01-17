"""
coil_fit.py
===========

Fit PF coil currents to a target boundary (LCFS) using coil Green's functions.

Three fitting modes
-------------------
(1) method="boundary_value"  (classic ridge regression on absolute psi)
    min_{I,c} ||A I + c*1 - b||^2 + λ||I||^2
  - Fits boundary psi to a specified target value b (often constant).
  - Includes optional offset c to handle psi gauge.
  - WARNING: if b is constant 0 and coils can produce psi=0 at I=0,
    the trivial solution I=0 is optimal and uninformative.

(2) method="contour"  (recommended for LCFS shaping; KKT + active-set clamp)
    min_I ||D (A I)||^2 + λ||I||^2   subject to  g^T I = psi_ref
  - D is a cyclic first-difference operator along boundary points.
  - This enforces "psi is (approximately) constant along the boundary" (i.e. boundary is a flux surface).
  - The constraint fixes the flux level to avoid the trivial I=0 solution.
    Common constraint: mean boundary psi equals psi_ref.
  - Bounds (if requested) are handled by a simple active-set clamp/refit loop.

(3) method="contour_qp"  (new; recommended default going forward)
    min_I ||D(AI)||^2 + λ||I||^2   subject to  g^T I = psi_ref,  I_lo <= I <= I_hi
  - Same contour objective as (2), but solved as a *convex constrained QP*
    using scipy.optimize.minimize(method="trust-constr"), with analytic
    gradient and constant Hessian.
  - Typically more robust than hand-rolled active-set for bounds, and
    produces reproducible results.

Conventions
-----------
• Grid:
    R: (NR,)
    Z: (NZ,)
    G_psi: (Nc, NZ, NR)   (coil index first, then [Z,R])
• Boundary points:
    boundary_pts: (Nb, 2)  with columns [R, Z]
• Interpolation:
    RegularGridInterpolator axes (Z, R), consistent with arrays shaped (NZ, NR).

Public API
----------
fit_pf_currents_to_boundary(
    G_psi,
    boundary_pts,
    R, Z,
    psi_target,
    reg_lambda,
    I_bounds=None,
    method="boundary_value",
    fit_offset=True,
    psi_ref=1.0,
    constraint="mean",
) -> dict

Minimum returned keys:
• I_fit
• residual_rms
• psi_boundary_fit

Plus diagnostics, depending on method.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import Bounds, LinearConstraint, minimize


ArrayLike = Union[np.ndarray, float, int]


# ============================================================
# INTERNAL: matrix building
# ============================================================

def _build_A_from_greens(
    G_psi: np.ndarray,
    boundary_pts: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
) -> np.ndarray:
    """Interpolate coil greens onto boundary points to form A (Nb, Nc)."""
    G_psi = np.asarray(G_psi, float)
    boundary_pts = np.asarray(boundary_pts, float)
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    if G_psi.ndim != 3:
        raise ValueError("G_psi must have shape (Nc, NZ, NR).")
    Nc, NZ, NR = G_psi.shape
    if R.size != NR or Z.size != NZ:
        raise ValueError("Grid mismatch between G_psi and R/Z.")
    if boundary_pts.ndim != 2 or boundary_pts.shape[1] != 2:
        raise ValueError("boundary_pts must have shape (Nb,2) with columns [R,Z].")

    Nb = boundary_pts.shape[0]
    pts_ZR = np.column_stack([boundary_pts[:, 1], boundary_pts[:, 0]])  # (Z, R)

    A = np.empty((Nb, Nc), dtype=float)
    for c in range(Nc):
        interp = RegularGridInterpolator(
            (Z, R),
            G_psi[c, :, :],
            bounds_error=False,
            fill_value=np.nan,
        )
        vals = interp(pts_ZR)
        if not np.all(np.isfinite(vals)):
            bad = np.where(~np.isfinite(vals))[0]
            raise ValueError(
                f"Boundary interpolation out of grid for coil {c}: {bad.size} invalid points.\n"
                f"Check boundary lies inside grid extents."
            )
        A[:, c] = vals
    return A


def _cyclic_first_difference_matrix(N: int) -> np.ndarray:
    """
    D psi has components psi[k+1]-psi[k] with cyclic wrap at end.
    Returns D shape (N, N).
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    D = np.zeros((N, N), dtype=float)
    for k in range(N):
        D[k, k] = -1.0
        D[k, (k + 1) % N] = 1.0
    return D


def _broadcast_bounds(Nc: int, I_bounds: Tuple[ArrayLike, ArrayLike]) -> Tuple[np.ndarray, np.ndarray]:
    lo_in, hi_in = I_bounds
    I_lo = np.full(Nc, float(lo_in), dtype=float) if np.isscalar(lo_in) else np.asarray(lo_in, float).reshape(-1)
    I_hi = np.full(Nc, float(hi_in), dtype=float) if np.isscalar(hi_in) else np.asarray(hi_in, float).reshape(-1)
    if I_lo.size != Nc or I_hi.size != Nc:
        raise ValueError("I_bounds arrays must have shape (Nc,) if not scalar.")
    if np.any(I_hi < I_lo):
        raise ValueError("Invalid I_bounds: some upper bounds are < lower bounds.")
    return I_lo, I_hi


def _build_contour_matrices(A: np.ndarray, reg_lambda: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For contour objectives, build:
      D (Nb,Nb), K=D@A (Nb,Nc), M0 = K^T K + λ I (Nc,Nc)
    """
    Nb, Nc = A.shape
    lam = float(reg_lambda)
    D = _cyclic_first_difference_matrix(Nb)
    K = D @ A
    M0 = (K.T @ K) + lam * np.eye(Nc, dtype=float)
    return D, K, M0


def _build_constraint_vector(A: np.ndarray, constraint: str) -> np.ndarray:
    """
    Build g so that the constraint is:
        g^T I = psi_ref
    from either:
      - "mean": mean boundary psi equals psi_ref
      - "point" or "point:k": psi at a boundary point equals psi_ref
    """
    Nb, Nc = A.shape
    constraint = str(constraint).lower().strip()
    if constraint == "mean":
        m = (1.0 / Nb) * np.ones(Nb, dtype=float)
        g = (m @ A).reshape(-1)  # (Nc,)
    elif constraint.startswith("point"):
        if ":" in constraint:
            idx = int(constraint.split(":", 1)[1])
        else:
            idx = 0
        idx = int(np.clip(idx, 0, Nb - 1))
        g = A[idx, :].copy()
    else:
        raise ValueError("constraint must be 'mean' or 'point' (optionally 'point:k').")
    return g.astype(float)


# ============================================================
# METHOD 1: boundary value ridge (with optional offset)
# ============================================================

def _solve_boundary_value(
    A: np.ndarray,
    b: np.ndarray,
    reg_lambda: float,
    I_bounds: Optional[Tuple[ArrayLike, ArrayLike]],
    fit_offset: bool,
) -> Dict[str, np.ndarray]:
    Nb, Nc = A.shape
    lam = float(reg_lambda)

    # Augment with offset column if requested: A_aug = [A, 1]
    if fit_offset:
        A_aug = np.hstack([A, np.ones((Nb, 1), dtype=float)])
        nvar = Nc + 1
    else:
        A_aug = A
        nvar = Nc

    AT = A_aug.T
    ATA = AT @ A_aug
    ATb = AT @ b

    # Regularize currents only, never offset
    reg = np.zeros(nvar, dtype=float)
    reg[:Nc] = lam
    M = ATA + np.diag(reg)

    x = np.linalg.solve(M, ATb)  # x=[I, offset] if fit_offset else [I]

    I_lo = None
    I_hi = None
    clamped = np.zeros(Nc, dtype=bool)

    if I_bounds is not None:
        I_lo, I_hi = _broadcast_bounds(Nc, I_bounds)

        # Active-set clamp loop on currents only; offset remains free
        for _ in range(30):
            x_prev = x.copy()

            x[:Nc] = np.minimum(np.maximum(x[:Nc], I_lo), I_hi)
            clamped = (x[:Nc] <= I_lo) | (x[:Nc] >= I_hi)
            free = ~clamped

            if not np.any(free):
                if fit_offset:
                    x[Nc] = float(np.mean(b - (A @ x[:Nc])))
                break

            if np.allclose(x, x_prev, rtol=0.0, atol=0.0):
                break

            # Solve for free currents (+ offset if present)
            solve_cols = list(np.where(free)[0])
            if fit_offset:
                solve_cols.append(Nc)

            all_cols = np.arange(nvar)
            fix_cols = np.array([c for c in all_cols if c not in solve_cols], dtype=int)

            A_solve = A_aug[:, solve_cols]
            A_fix = A_aug[:, fix_cols]
            x_fix = x[fix_cols]
            b_eff = b - (A_fix @ x_fix if A_fix.size else 0.0)

            Ms = (A_solve.T @ A_solve)
            reg_s = np.zeros(len(solve_cols), dtype=float)
            for ii, col in enumerate(solve_cols):
                reg_s[ii] = lam if col < Nc else 0.0
            Ms = Ms + np.diag(reg_s)

            rhs = A_solve.T @ b_eff
            x_solve = np.linalg.solve(Ms, rhs)

            for ii, col in enumerate(solve_cols):
                x[col] = x_solve[ii]

            # stable clamp?
            x_next = x.copy()
            x_next[:Nc] = np.minimum(np.maximum(x_next[:Nc], I_lo), I_hi)
            clamped_next = (x_next[:Nc] <= I_lo) | (x_next[:Nc] >= I_hi)
            if np.array_equal(clamped_next, clamped):
                x = x_next
                clamped = clamped_next
                break

        x[:Nc] = np.minimum(np.maximum(x[:Nc], I_lo), I_hi)
        clamped = (x[:Nc] <= I_lo) | (x[:Nc] >= I_hi)
        if fit_offset:
            x[Nc] = float(np.mean(b - (A @ x[:Nc])))

    if fit_offset:
        I = x[:Nc].copy()
        offset = float(x[Nc])
        psi_fit = (A @ I) + offset
    else:
        I = x[:Nc].copy()
        offset = 0.0
        psi_fit = A @ I

    residual = psi_fit - b
    residual_rms = float(np.sqrt(np.mean(residual * residual)))

    return {
        "I_fit": I.astype(float),
        "offset": np.array(offset, dtype=float),
        "psi_boundary_fit": psi_fit.astype(float),
        "residual": residual.astype(float),
        "residual_rms": np.array(residual_rms, dtype=float),
        "clamped": clamped.astype(bool),
        **({} if I_lo is None else {"I_lo": I_lo.astype(float), "I_hi": I_hi.astype(float)}),
    }


# ============================================================
# METHOD 2: contour fit (KKT + active-set clamp/refit)
# ============================================================

def _solve_contour(
    A: np.ndarray,
    reg_lambda: float,
    psi_ref: float,
    constraint: str,
    I_bounds: Optional[Tuple[ArrayLike, ArrayLike]],
) -> Dict[str, np.ndarray]:
    """
    Solve:
        min_I ||D(AI)||^2 + λ||I||^2   s.t.  g^T I = psi_ref
    """
    Nb, Nc = A.shape
    lam = float(reg_lambda)

    D, K, M0 = _build_contour_matrices(A, reg_lambda=lam)
    g = _build_constraint_vector(A, constraint=constraint)

    def _solve_kkt(Mm: np.ndarray, gg: np.ndarray, rhs_c: float) -> np.ndarray:
        """
        Solve:
            [Mm  gg] [x ] = [0]
            [gg^T 0] [mu]   [rhs_c]
        Returns x.
        """
        gg = np.asarray(gg, dtype=float).reshape(-1)
        n = int(Mm.shape[0])
        if Mm.shape != (n, n):
            raise ValueError("Mm must be square.")
        if gg.size != n:
            raise ValueError(f"gg length mismatch: expected {n}, got {gg.size}")

        KKT = np.zeros((n + 1, n + 1), dtype=float)
        KKT[:n, :n] = Mm
        KKT[:n, n] = gg
        KKT[n, :n] = gg

        rhs = np.zeros(n + 1, dtype=float)
        rhs[n] = float(rhs_c)

        sol = np.linalg.solve(KKT, rhs)
        return sol[:n]

    # Unbounded KKT solution as starting point
    I = _solve_kkt(M0, g, float(psi_ref))

    I_lo = None
    I_hi = None
    clamped = np.zeros(Nc, dtype=bool)

    # Bounds via active-set clamp/refit on currents
    if I_bounds is not None:
        I_lo, I_hi = _broadcast_bounds(Nc, I_bounds)

        for _ in range(50):
            I_prev = I.copy()

            I = np.minimum(np.maximum(I, I_lo), I_hi)
            eps = 1e-9
            clamped = (I <= I_lo + eps) | (I >= I_hi - eps)
            free = ~clamped

            if not np.any(free):
                if abs(float(g @ I) - float(psi_ref)) > 1e-9 * max(1.0, abs(psi_ref)):
                    raise RuntimeError("Contour fit infeasible under bounds (constraint cannot be satisfied).")
                break

            Af = A[:, free]
            gf = g[free]
            rhs_c = float(psi_ref - (g[clamped] @ I[clamped]))

            Df = D
            Kf = Df @ Af
            Mf = (Kf.T @ Kf) + lam * np.eye(Af.shape[1], dtype=float)

            If = _solve_kkt(Mf, gf, rhs_c)
            I[free] = If

            I_next = np.minimum(np.maximum(I, I_lo), I_hi)
            clamped_next = (I_next <= I_lo) | (I_next >= I_hi)
            if np.array_equal(clamped_next, clamped):
                I = I_next
                clamped = clamped_next
                break

            if np.linalg.norm(I - I_prev) <= 0.0:
                break

        I = np.minimum(np.maximum(I, I_lo), I_hi)
        clamped = (I <= I_lo) | (I >= I_hi)

    psi_fit = A @ I
    dpsi = (_cyclic_first_difference_matrix(Nb) @ psi_fit)
    contour_rms = float(np.sqrt(np.mean(dpsi * dpsi)))

    b = np.full(Nb, float(psi_ref), dtype=float)
    residual = psi_fit - b

    return {
        "I_fit": I.astype(float),
        "offset": np.array(0.0, dtype=float),
        "psi_boundary_fit": psi_fit.astype(float),
        "residual": residual.astype(float),
        "residual_rms": np.array(contour_rms, dtype=float),
        "contour_rms": np.array(contour_rms, dtype=float),
        "clamped": clamped.astype(bool),
        "psi_ref": np.array(float(psi_ref), dtype=float),
        "constraint": np.array(str(constraint), dtype="S"),
        **({} if I_lo is None else {"I_lo": I_lo.astype(float), "I_hi": I_hi.astype(float)}),
    }


# ============================================================
# METHOD 3: contour QP (trust-constr)  <-- NEW
# ============================================================

def _solve_contour_qp(
    A: np.ndarray,
    reg_lambda: float,
    psi_ref: float,
    constraint: str,
    I_bounds: Optional[Tuple[ArrayLike, ArrayLike]],
) -> Dict[str, np.ndarray]:
    """
    Solve the convex constrained QP:
        min_I  ||D(AI)||^2 + λ||I||^2
        s.t.   g^T I = psi_ref
               I_lo <= I <= I_hi   (optional)

    Uses scipy.optimize.minimize(method="trust-constr") with analytic gradient and
    constant Hessian (quadratic objective).
    """
    Nb, Nc = A.shape
    lam = float(reg_lambda)

    D, K, M0 = _build_contour_matrices(A, reg_lambda=lam)
    g = _build_constraint_vector(A, constraint=constraint)

    # Objective: f(I) = I^T M0 I  (since ||K I||^2 + lam||I||^2)
    # Gradient:  ∇f = 2 M0 I
    # Hessian:   ∇²f = 2 M0  (constant)
    H = 2.0 * M0

    def fun(I: np.ndarray) -> float:
        I = np.asarray(I, float).reshape(-1)
        return float(I @ (M0 @ I))

    def jac(I: np.ndarray) -> np.ndarray:
        I = np.asarray(I, float).reshape(-1)
        return (2.0 * (M0 @ I)).astype(float)

    def hess(_I: np.ndarray) -> np.ndarray:
        return H

    # Equality constraint g^T I = psi_ref
    lin_con = LinearConstraint(g.reshape(1, -1), np.array([psi_ref], float), np.array([psi_ref], float))

    # Bounds (optional)
    if I_bounds is not None:
        I_lo, I_hi = _broadcast_bounds(Nc, I_bounds)
        bnds = Bounds(I_lo, I_hi)
    else:
        bnds = None
        I_lo = None
        I_hi = None

    # Starting point: use the (unbounded) KKT contour solution, then clip into bounds.
    # This gives a very good initial guess and usually converges in few iterations.
    try:
        I0 = _solve_contour(A, reg_lambda=lam, psi_ref=psi_ref, constraint=constraint, I_bounds=None)["I_fit"]
    except Exception:
        I0 = np.zeros(Nc, dtype=float)

    if bnds is not None:
        I0 = np.minimum(np.maximum(I0, I_lo), I_hi)

    # Ensure equality constraint is satisfied at start as best-effort:
    # Project along g if possible (only if g is not ~0).
    gg = float(g @ g)
    if gg > 0.0:
        # shift I0 by alpha*g to satisfy g^T I0 = psi_ref (then bounds may break it)
        alpha = (float(psi_ref) - float(g @ I0)) / gg
        I0 = I0 + alpha * g
        if bnds is not None:
            I0 = np.minimum(np.maximum(I0, I_lo), I_hi)

    res = minimize(
        fun,
        I0,
        method="trust-constr",
        jac=jac,
        hess=hess,
        constraints=[lin_con],
        bounds=bnds,
        options={
            "verbose": 0,
            "gtol": 1e-10,
            "xtol": 1e-12,
            "barrier_tol": 1e-12,
            "maxiter": 2000,
        },
    )

    if not res.success:
        raise RuntimeError(f"contour_qp failed: {res.message}")

    I = np.asarray(res.x, float).reshape(-1)

    clamped = np.zeros(Nc, dtype=bool)
    if I_bounds is not None:
        eps = 1e-9
        clamped = (I <= I_lo + eps) | (I >= I_hi - eps)

        # Feasibility check for the equality constraint (trust-constr should respect it tightly).
        # If bounds make it infeasible, SciPy may fail; we already raise above if so.
        if abs(float(g @ I) - float(psi_ref)) > 1e-6 * max(1.0, abs(psi_ref)):
            raise RuntimeError("contour_qp returned solution that violates constraint unexpectedly.")

    psi_fit = A @ I
    dpsi = D @ psi_fit
    contour_rms = float(np.sqrt(np.mean(dpsi * dpsi)))

    b = np.full(Nb, float(psi_ref), dtype=float)
    residual = psi_fit - b

    return {
        "I_fit": I.astype(float),
        "offset": np.array(0.0, dtype=float),
        "psi_boundary_fit": psi_fit.astype(float),
        "residual": residual.astype(float),
        # residual_rms is interpreted as contour_rms for contour methods
        "residual_rms": np.array(contour_rms, dtype=float),
        "contour_rms": np.array(contour_rms, dtype=float),
        "clamped": clamped.astype(bool),
        "psi_ref": np.array(float(psi_ref), dtype=float),
        "constraint": np.array(str(constraint), dtype="S"),
        "qp_success": np.array(bool(res.success), dtype=np.uint8),
        "qp_niter": np.array(int(res.niter), dtype=int),
        **({} if I_bounds is None else {"I_lo": I_lo.astype(float), "I_hi": I_hi.astype(float)}),
    }


# ============================================================
# PUBLIC API
# ============================================================

def fit_pf_currents_to_boundary(
    G_psi: np.ndarray,
    boundary_pts: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    psi_target: ArrayLike,
    reg_lambda: float,
    I_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    *,
    method: str = "boundary_value",
    fit_offset: bool = True,
    psi_ref: float = 1.0,
    constraint: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Fit PF coil currents.

    method="boundary_value":
      Fits absolute boundary psi (optionally with offset).
      Uses psi_target as b.

    method in ("contour", "contour_qp", ...):
      Fits boundary to be a constant-psi contour using first differences + a constraint.
      Uses psi_ref and constraint, ignores psi_target (except for logging by caller).

    Returns a dict with at least:
      I_fit, residual_rms, psi_boundary_fit
    plus diagnostics.
    """
    A = _build_A_from_greens(G_psi, boundary_pts, R, Z)
    Nb, _Nc = A.shape

    method = str(method).lower().strip()

    # --- boundary-value (ridge) ---
    if method in ("boundary_value", "ridge", "tikhonov"):
        if np.isscalar(psi_target):
            b = np.full(Nb, float(psi_target), dtype=float)
        else:
            b = np.asarray(psi_target, float).reshape(-1)
            if b.size != Nb:
                raise ValueError(f"psi_target array must have shape (Nb,), got {b.shape} for Nb={Nb}.")

        out = _solve_boundary_value(A, b, reg_lambda, I_bounds, fit_offset=fit_offset)
        psi_fit = out["psi_boundary_fit"]
        out.update({
            "A": A.astype(float),
            "b": b.astype(float),
            "psi_boundary_std": np.array(float(np.std(psi_fit)), dtype=float),
            "psi_boundary_ptp": np.array(float(np.ptp(psi_fit)), dtype=float),
            "method": np.array("boundary_value", dtype="S"),
        })
        return out

    # --- contour (active-set) ---
    if method in ("contour", "flux_surface", "lcfs"):
        out = _solve_contour(A, reg_lambda, psi_ref=float(psi_ref), constraint=str(constraint), I_bounds=I_bounds)
        psi_fit = out["psi_boundary_fit"]
        out.update({
            "A": A.astype(float),
            "b": np.full(Nb, float(psi_ref), dtype=float),
            "psi_boundary_std": np.array(float(np.std(psi_fit)), dtype=float),
            "psi_boundary_ptp": np.array(float(np.ptp(psi_fit)), dtype=float),
            "method": np.array("contour", dtype="S"),
        })
        return out

    # --- contour QP (trust-constr) ---
    if method in ("contour_qp", "qp", "contour-trust", "contour_trust"):
        out = _solve_contour_qp(A, reg_lambda, psi_ref=float(psi_ref), constraint=str(constraint), I_bounds=I_bounds)
        psi_fit = out["psi_boundary_fit"]
        out.update({
            "A": A.astype(float),
            "b": np.full(Nb, float(psi_ref), dtype=float),
            "psi_boundary_std": np.array(float(np.std(psi_fit)), dtype=float),
            "psi_boundary_ptp": np.array(float(np.ptp(psi_fit)), dtype=float),
            "method": np.array("contour_qp", dtype="S"),
        })
        return out

    raise ValueError(f"Unknown method '{method}'. Use 'boundary_value', 'contour', or 'contour_qp'.")


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing coil_fit.py (contour_qp method)")

    rng = np.random.default_rng(0)

    NR, NZ = 60, 50
    R = np.linspace(1.0, 2.2, NR)
    Z = np.linspace(-0.8, 0.8, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")

    Nb = 200
    th = np.linspace(0, 2 * np.pi, Nb, endpoint=False)
    boundary = np.column_stack([1.6 + 0.25 * np.cos(th), 0.0 + 0.35 * np.sin(th)])

    Nc = 8
    G_psi = np.zeros((Nc, NZ, NR), dtype=float)
    for c in range(Nc):
        a0, aR, aZ, aR2, aZ2 = rng.normal(size=5)
        wR, wZ = rng.uniform(0.5, 3.0, size=2)
        phase = rng.uniform(0, 2 * np.pi)
        G_psi[c] = (
            a0
            + aR * (RR - 1.5)
            + aZ * ZZ
            + aR2 * (RR - 1.5) ** 2
            + aZ2 * ZZ ** 2
            + 0.3 * np.sin(wR * (RR - 1.0) + wZ * ZZ + phase)
        )

    # Bounds in "physical I" here just as a test
    Imax = np.full(Nc, 2.0, dtype=float)
    I_bounds = (-Imax, +Imax)

    psi_ref = 1.0
    res = fit_pf_currents_to_boundary(
        G_psi=G_psi,
        boundary_pts=boundary,
        R=R, Z=Z,
        psi_target=0.0,          # ignored in contour modes
        reg_lambda=1e-6,
        I_bounds=I_bounds,
        method="contour_qp",
        psi_ref=psi_ref,
        constraint="mean",
    )

    I = np.asarray(res["I_fit"], float)
    psi_fit = np.asarray(res["psi_boundary_fit"], float)
    mean_psi = float(np.mean(psi_fit))
    contour_rms = float(res["contour_rms"])

    print("  ||I||:", float(np.linalg.norm(I)))
    print("  mean(psi_fit):", mean_psi, "target:", psi_ref)
    print("  contour_rms:", contour_rms)
    print("  clamped:", int(np.sum(res["clamped"])), "/", Nc)
    print("  qp_niter:", int(np.asarray(res.get("qp_niter", 0)).item()))

    assert np.linalg.norm(I) > 0.0, "Contour QP returned trivial zero currents."
    assert abs(mean_psi - psi_ref) < 1e-6, "Mean-psi constraint not satisfied (tolerance)."
    assert np.isfinite(contour_rms)

    print("coil_fit.py self-test passed.")