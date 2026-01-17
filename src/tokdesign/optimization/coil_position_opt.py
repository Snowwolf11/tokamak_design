"""
coil_position_opt.py
====================

Outer-loop optimization of PF coil positions (centers) to improve boundary
flux-surface fit performance.

This module does NOT replace the inner current fit. It wraps it:

  positions p  -> build A(p) on boundary -> fit currents I*(p) -> score(p)

The inner solve is convex and fast (e.g. contour_qp). The outer problem is
nonconvex and solved with a derivative-free optimizer (SciPy differential evolution).

Key design choices
------------------
• Keep coil_fit.py as the "inner problem" solver.
• Optimize only coil centers (R,Z). Radii are held fixed (engineering).
• Use boundary-only evaluation:
    - Build A directly at boundary points (Nb x Nc) rather than recomputing full
      grid greens for each candidate.
  This makes position search feasible.

Return values include:
• best centers
• best inner-fit results
• history of evaluated candidates (useful for debugging/tuning)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution


@dataclass
class PositionOptConfig:
    enabled: bool = False
    max_iter: int = 80
    popsize: int = 18                 # SciPy DE uses popsize * dim population
    seed: int = 0
    polish: bool = True
    tol: float = 0.0                  # DE termination tol; 0 disables early stop by tol
    atol: float = 0.0
    # objective weights
    w_contour: float = 1.0
    w_xnorm: float = 0.02
    w_clamp: float = 2.0
    w_move: float = 0.1
    # cache resolution (meters). Round centers for cache key.
    cache_round_m: float = 1e-3
    # Progress printing
    progress: bool = True
    progress_every: int = 25


def _as_float(x) -> float:
    return float(np.asarray(x).item())


def optimize_coil_positions(
    *,
    centers0: np.ndarray,                 # (Nc,2)
    movable: Sequence[int],               # indices into coils
    bounds_RZ: Sequence[Tuple[float, float, float, float]],  # per movable coil: (Rmin,Rmax,Zmin,Zmax)
    build_A: Callable[[np.ndarray], np.ndarray],             # centers -> A(Nb,Nc)
    solve_inner: Callable[[np.ndarray], Dict[str, np.ndarray]],  # A -> inner fit dict (must contain I_fit, contour_rms, clamped)
    cfg: PositionOptConfig,
    I_max: Optional[np.ndarray] = None,   # (Nc,) optional; used to compute x_norm if not present
) -> Dict[str, object]:
    """
    Optimize coil centers (R,Z) for selected coils.

    Parameters
    ----------
    centers0:
        Baseline coil centers, shape (Nc,2)
    movable:
        Coil indices that are allowed to move.
    bounds_RZ:
        Per-movable coil bounds (Rmin,Rmax,Zmin,Zmax).
    build_A:
        Function that returns A for given centers.
        NOTE: should keep coil ordering identical to centers0.
    solve_inner:
        Function that solves the inner current fit given A.
        Should return at least:
          - I_fit (Nc,)
          - contour_rms (scalar)   OR residual_rms meaning contour_rms
          - clamped (Nc,) bool
    cfg:
        Position optimization config + weights.
    I_max:
        Optional coil current limits for x-norm metric.

    Returns
    -------
    dict with:
      best_centers, best_score, best_fit, history (arrays), n_eval
    """
    centers0 = np.asarray(centers0, float)
    if centers0.ndim != 2 or centers0.shape[1] != 2:
        raise ValueError("centers0 must have shape (Nc,2).")
    Nc = centers0.shape[0]

    movable = [int(i) for i in movable]
    if len(movable) == 0:
        raise ValueError("No movable coils provided.")
    if any(i < 0 or i >= Nc for i in movable):
        raise ValueError("movable indices out of range.")
    if len(bounds_RZ) != len(movable):
        raise ValueError("bounds_RZ must have same length as movable.")

    # Flatten parameter vector p = [R_m0, Z_m0, R_m1, Z_m1, ...]
    p0 = []
    de_bounds: List[Tuple[float, float]] = []
    for (i, (Rmin, Rmax, Zmin, Zmax)) in zip(movable, bounds_RZ):
        p0.extend([centers0[i, 0], centers0[i, 1]])
        de_bounds.extend([(float(Rmin), float(Rmax)), (float(Zmin), float(Zmax))])
    p0 = np.array(p0, dtype=float)

    # Cache: key -> (score, fit, centers)
    cache: Dict[Tuple[float, ...], Tuple[float, Dict[str, np.ndarray], np.ndarray]] = {}

    # History storage (append; later stacked)
    hist_p: List[np.ndarray] = []
    hist_score: List[float] = []
    hist_contour: List[float] = []
    hist_xnorm: List[float] = []
    hist_clamp: List[int] = []
    hist_move: List[float] = []

    #Progress / best so far
    eval_count = 0
    best_seen_score = np.inf
    best_seen_contour = np.inf
    best_seen_clamp = 0

    def _round_key(centers: np.ndarray) -> Tuple[float, ...]:
        r = float(cfg.cache_round_m)
        if r <= 0:
            return tuple(centers.flatten().tolist())
        return tuple(np.round(centers.flatten() / r) * r)

    def _assemble_centers(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float).reshape(-1)
        centers = centers0.copy()
        for k, idx in enumerate(movable):
            centers[idx, 0] = p[2 * k + 0]
            centers[idx, 1] = p[2 * k + 1]
        return centers

    def _score_from_fit(fit: Dict[str, np.ndarray], centers: np.ndarray) -> Tuple[float, float, float, int, float]:
        # Contour metric:
        if "contour_rms" in fit:
            contour = _as_float(fit["contour_rms"])
        else:
            contour = _as_float(fit["residual_rms"])  # by convention, contour methods report this

        I_fit = np.asarray(fit["I_fit"], float).reshape(-1)
        if I_fit.size != Nc:
            raise ValueError("Inner fit returned wrong I_fit size.")
        clamped = np.asarray(fit.get("clamped", np.zeros(Nc, dtype=bool)), bool).reshape(-1)
        clamp_count = int(np.sum(clamped))

        # x_norm: prefer to use x-fit if provided; otherwise compute from I_max
        if "x_fit" in fit:
            x = np.asarray(fit["x_fit"], float).reshape(-1)
        elif I_max is not None:
            I_max_arr = np.asarray(I_max, float).reshape(-1)
            x = I_fit / np.maximum(I_max_arr, 1e-30)
        else:
            # fallback: normalize by max abs current to get a scale-free-ish metric
            denom = max(1e-30, float(np.max(np.abs(I_fit))))
            x = I_fit / denom
        xnorm = float(np.dot(x, x))

        # Move penalty: mean squared displacement of movable coils from baseline (meters^2)
        d = centers[movable, :] - centers0[movable, :]
        move = float(np.mean(np.sum(d * d, axis=1))) if d.size else 0.0

        score = (
            float(cfg.w_contour) * contour
            + float(cfg.w_xnorm) * xnorm
            + float(cfg.w_clamp) * (clamp_count / max(1, Nc))
            + float(cfg.w_move) * move
        )
        return score, contour, xnorm, clamp_count, move

    def objective(p: np.ndarray) -> float:
        nonlocal eval_count, best_seen_score, best_seen_contour, best_seen_clamp

        eval_count += 1

                
        centers = _assemble_centers(p)
        key = _round_key(centers)

        if key in cache:
            score, _fit, _centers = cache[key]
            #Even if cached its useful to show progress occasionally
            if cfg.progress and (eval_count % max(1,int(cfg.progress_every)) == 0):
                print(f"[pos-opt] eval {eval_count:6d}: cached_score={score:.6g}   best={best_seen_score}")
            return float(score)

        A = build_A(centers)
        fit = solve_inner(A)
        score, contour, xnorm, clamp_count, move = _score_from_fit(fit, centers)

        cache[key] = (float(score), fit, centers.copy())

        # History append (store actual p used)
        hist_p.append(np.asarray(p, float).copy())
        hist_score.append(float(score))
        hist_contour.append(float(contour))
        hist_xnorm.append(float(xnorm))
        hist_clamp.append(int(clamp_count))
        hist_move.append(float(move))

        # Update best-so-far
        if score < best_seen_score:
            best_seen_score = float(score)
            best_seen_contour = float(contour)
            best_seen_clamp = float(clamp_count)

        if cfg.progress and (eval_count % max(1,int(cfg.progress_every)) == 0):
            print(
                "[pos-opt] eval {n:6d}: score = {s:.6g}  contour={c:.6g}  clamps={k:d}  "
                "xnorm={x:.3g}  move_m2={m:.3g}  best={bs:.6g}".format(
                    n=eval_count, s=score, c=contour, k=clamp_count,
                    x=xnorm, m=move, bs = best_seen_score
                )
            )
        return float(score)

    def _de_callback(xk: np.ndarray, convergence: float) -> bool:
        #Return True to stop early (we dont stop here).
        if cfg.progress:
            print(f"[pos-out] generation callback: conv={convergence:.3g}  best={best_seen_score:.6g}")
        return False
    
    # Run SciPy differential evolution
    result = differential_evolution(
        objective,
        bounds=de_bounds,
        maxiter=int(cfg.max_iter),
        popsize=int(cfg.popsize),
        seed=int(cfg.seed),
        polish=bool(cfg.polish),
        tol=float(cfg.tol),
        atol=float(cfg.atol),
        updating="deferred",
        workers=1,  # keep deterministic & simple; can be >1 later
        disp=False,
        callback=_de_callback
    )

    p_best = np.asarray(result.x, float)
    centers_best = _assemble_centers(p_best)
    key_best = _round_key(centers_best)
    if key_best not in cache:
        # Shouldn't happen, but safe:
        A_best = build_A(centers_best)
        fit_best = solve_inner(A_best)
        score_best, *_ = _score_from_fit(fit_best, centers_best)
    else:
        score_best, fit_best, _ = cache[key_best]

    # Pack history arrays
    hist = {
        "p": np.vstack(hist_p).astype(float) if hist_p else np.zeros((0, len(p0)), float),
        "score": np.asarray(hist_score, float),
        "contour_rms": np.asarray(hist_contour, float),
        "x_norm": np.asarray(hist_xnorm, float),
        "clamp_count": np.asarray(hist_clamp, int),
        "move_m2": np.asarray(hist_move, float),
    }

    return {
        "best_centers": centers_best.astype(float),
        "best_score": float(score_best),
        "best_fit": fit_best,
        "history": hist,
        "n_eval": int(len(hist_score)),
        "de_success": bool(result.success),
        "de_message": str(result.message),
        "de_nit": int(result.nit),
    }