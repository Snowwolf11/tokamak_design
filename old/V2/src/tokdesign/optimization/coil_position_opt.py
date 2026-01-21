#!/usr/bin/env python3
"""
coil_position_opt.py
====================

Outer-loop *coil position* optimizer for PF shaping.

This module is intentionally "optimization-only":
  • It does NOT own any physics primitives (no GS solve here).
  • It does NOT decide how psi is computed (analytic loops vs precomputed greens).
  • It only coordinates an outer geometry search and scores candidates using
    outputs from an inner "fit currents" solve.

You provide two callables:
  build_A(centers) -> A
      Builds the boundary influence matrix for a given coil geometry.
      The caller decides whether A is built from analytic loop psi or other methods.

  solve_inner(A) -> fit_dict
      Solves the inner current-fit problem for that A and returns diagnostics.

Critical design rule (fixes the "tiny xnorm" bug)
-------------------------------------------------
The outer loop must be told what the inner variable represents:
  • var_mode="x"  : inner variable is x = I/I_max (A columns are already scaled if desired)
  • var_mode="I"  : inner variable is physical current in A

In var_mode="x", xnorm must be computed from x directly.
In var_mode="I", xnorm can be computed as (I/I_max) if you pass I_max.

This file enforces consistent xnorm handling and avoids "double normalization".

Vessel avoidance (hard keep-out)
--------------------------------
Optional hard constraint to prevent coils (including finite radius a + clearance)
from intersecting the vessel.

Design choice:
  • No logging here: stage scripts handle warnings.
  • No HDF5 I/O here: stage scripts read vessel boundary + coil radii and pass them in.
  • If vessel avoidance is enabled but the *initial* positions violate, the constraint
    is disabled for this run and the violating indices are returned to the caller.

Return format
-------------
optimize_coil_positions returns a dict with:
  best_centers: (Nc,2)
  best_score: float
  best_fit: dict (as returned by solve_inner + a few extras)
  history: dict of arrays for plotting/debugging
  de_success, de_message, de_nit, n_eval
  vessel_avoidance: diagnostics about vessel constraint usage / initial violations

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from tokdesign.optimization.vessel_avoidance import (
    VesselAvoidanceConfig,
    vessel_violation_mask,
    violating_indices,
)

Array = np.ndarray


@dataclass
class PositionOptConfig:
    """
    Configuration for the outer (position) optimization.

    Weights:
      score = w_contour*contour_rms
            + w_xnorm*x_norm2
            + w_clamp*(clamp_count/Nc)
            + w_move*move_m2
    """
    enabled: bool = True

    # Differential evolution settings
    max_iter: int = 80
    popsize: int = 18
    seed: int = 0
    polish: bool = True
    tol: float = 0.0
    atol: float = 0.0

    # Objective weights
    w_contour: float = 1.0
    w_xnorm: float = 0.02
    w_clamp: float = 2.0
    w_move: float = 0.1

    # Caching: round coil centers to this many meters when hashing candidates
    cache_round_m: float = 1e-3

    # Logging/progress
    verbose: bool = True
    print_every: int = 25  # evaluations


def optimize_coil_positions(
    *,
    centers0: Array,
    movable: Sequence[int],
    bounds_RZ: Sequence[Tuple[float, float, float, float]],
    build_A: Callable[[Array], Array],
    solve_inner: Callable[[Array], Dict[str, Any]],
    cfg: PositionOptConfig,
    var_mode: str = "x",
    I_max: Optional[Array] = None,
    # ---- NEW: vessel avoidance (data passed in from stage scripts) ----
    vessel_avoid: Optional[VesselAvoidanceConfig] = None,
    vessel_boundary: Optional[Array] = None,  # (Nv,2) in (R,Z)
    coil_a: Optional[Array] = None,           # scalar or (Nc,)
) -> Dict[str, Any]:
    """
    Optimize coil centers (R,Z) for a subset of coils.

    Parameters
    ----------
    centers0:
        Baseline coil centers (Nc,2).

    movable:
        Indices of coils that may be moved.

    bounds_RZ:
        Per-movable bounds: [(Rmin,Rmax,Zmin,Zmax), ...] length == len(movable).

    build_A(centers):
        Builds boundary response matrix for current fit.

    solve_inner(A):
        Inner solver. Must return at least:
          "I_fit": (Nc,)
        and ideally:
          "contour_rms" (float) for contour methods OR "residual_rms" (float) fallback
          "clamped": (Nc,) bool  (optional)
          "x_fit": (Nc,) if var_mode="x" (strongly recommended)

    cfg:
        Outer optimizer config.

    var_mode:
        "x" -> inner variable is x=I/I_max (preferred when using limit-weighting)
        "I" -> inner variable is I (A per-amp columns, bounds in A)

    I_max:
        Required only if var_mode="I" AND you want x_norm2 computed as (I/I_max)^2.
        Must NOT be passed if var_mode="x" (to avoid double normalization mistakes).

    vessel_avoid:
        VesselAvoidanceConfig(enabled, clearance). If enabled, the optimizer hard-rejects
        candidate positions that violate the constraint.

    vessel_boundary:
        Vessel polygon/polyline in (R,Z), shape (Nv,2). Required if vessel_avoid.enabled.

    coil_a:
        Coil radius a [m] as scalar or (Nc,) array. Required if vessel_avoid.enabled.

    Returns
    -------
    dict with keys:
      best_centers, best_score, best_fit, history, de_success, de_message, de_nit, n_eval,
      vessel_avoidance (dict with applied flag, clearance, initial_violations list)
    """

    centers0 = np.asarray(centers0, float)
    if centers0.ndim != 2 or centers0.shape[1] != 2:
        raise ValueError("centers0 must have shape (Nc,2)")
    Nc = centers0.shape[0]

    movable = list(movable)
    if len(movable) != len(bounds_RZ):
        raise ValueError("bounds_RZ length must match movable length")

    var_mode = str(var_mode).strip().lower()
    if var_mode not in ("x", "i"):
        raise ValueError("var_mode must be 'x' or 'I'")

    if var_mode == "x" and I_max is not None:
        # Strong guardrail: passing I_max here is exactly how the old bug happened.
        raise ValueError(
            "I_max must be None when var_mode='x' (inner variable already normalized)."
        )

    if var_mode == "i":
        if I_max is not None:
            I_max = np.asarray(I_max, float).reshape(-1)
            if I_max.size != Nc:
                raise ValueError("I_max must have size Nc")
            if np.any(~np.isfinite(I_max)) or np.any(I_max <= 0):
                raise ValueError("I_max must be finite and >0")
        # I_max can still be None; then x_norm2 falls back to ||I||^2, not ideal but safe.

    # ---- NEW: vessel avoidance setup + initial feasibility check (no logging here) ----
    vessel_avoid = vessel_avoid or VesselAvoidanceConfig(enabled=False, clearance=0.0)
    initial_vessel_violations: List[int] = []
    vessel_avoid_applied = bool(vessel_avoid.enabled)

    if vessel_avoid.enabled:
        if vessel_boundary is None or coil_a is None:
            raise ValueError("vessel_avoid.enabled=True but vessel_boundary or coil_a not provided")

        initial_vessel_violations = violating_indices(
            vessel_boundary=vessel_boundary,
            centers=centers0,
            coil_a=coil_a,
            clearance=float(vessel_avoid.clearance),
            movable=movable,
        )

        # Per requirement: if initial violates, disable constraint for this run.
        if initial_vessel_violations:
            vessel_avoid = VesselAvoidanceConfig(enabled=False, clearance=float(vessel_avoid.clearance))
            vessel_avoid_applied = False

    # ---- helpers to map p <-> centers ----
    bounds_de: List[Tuple[float, float]] = []
    for (Rmin, Rmax, Zmin, Zmax) in bounds_RZ:
        bounds_de.append((float(Rmin), float(Rmax)))
        bounds_de.append((float(Zmin), float(Zmax)))

    def _assemble_centers(p: Array) -> Array:
        c = centers0.copy()
        p = np.asarray(p, float).reshape(-1)
        for j, idx in enumerate(movable):
            c[idx, 0] = p[2 * j + 0]
            c[idx, 1] = p[2 * j + 1]
        return c

    # ---- scoring ----
    def _contour_metric(fit: Dict[str, Any]) -> float:
        # Prefer contour_rms if present; else residual_rms; else infer from residual.
        if "contour_rms" in fit and fit["contour_rms"] is not None:
            return float(np.asarray(fit["contour_rms"]).item())
        if "residual_rms" in fit and fit["residual_rms"] is not None:
            return float(np.asarray(fit["residual_rms"]).item())
        if "residual" in fit:
            r = np.asarray(fit["residual"], float).reshape(-1)
            return float(np.sqrt(np.mean(r * r)))
        raise KeyError("Inner fit dict must contain contour_rms or residual_rms or residual")

    def _x_vector(fit: Dict[str, Any]) -> Array:
        """Return x used for x_norm2 term, with strict consistency across var modes."""
        I_fit = np.asarray(fit["I_fit"], float).reshape(-1)
        if I_fit.size != Nc:
            raise ValueError(f"I_fit size mismatch: expected {Nc}, got {I_fit.size}")

        if var_mode == "x":
            # Inner variable is already x; prefer explicit x_fit if given.
            if "x_fit" in fit:
                x = np.asarray(fit["x_fit"], float).reshape(-1)
                if x.size != Nc:
                    raise ValueError("x_fit size mismatch")
                return x
            return I_fit  # treat I_fit as x (convention)
        else:
            # var_mode == "I"
            if I_max is not None:
                return I_fit / I_max
            # Fallback: no normalization info; use I directly.
            return I_fit

    def _clamp_count(fit: Dict[str, Any]) -> int:
        if "clamped" in fit and fit["clamped"] is not None:
            cl = np.asarray(fit["clamped"]).astype(bool).reshape(-1)
            if cl.size != Nc:
                return int(np.sum(cl[:Nc]))
            return int(np.sum(cl))
        return 0

    def _move_m2(centers: Array) -> float:
        if len(movable) == 0:
            return 0.0
        d2 = []
        for idx in movable:
            dR = float(centers[idx, 0] - centers0[idx, 0])
            dZ = float(centers[idx, 1] - centers0[idx, 1])
            d2.append(dR * dR + dZ * dZ)
        return float(np.mean(d2))

    # ---- caching by rounded centers ----
    cache: Dict[Tuple[float, ...], Dict[str, Any]] = {}
    round_m = float(cfg.cache_round_m)

    def _cache_key(centers: Array) -> Tuple[float, ...]:
        # Key only on movable coils, because non-movable are constant.
        v: List[float] = []
        for idx in movable:
            v.append(float(np.round(centers[idx, 0] / round_m) * round_m))
            v.append(float(np.round(centers[idx, 1] / round_m) * round_m))
        return tuple(v)

    # ---- history ----
    hist_p: List[Array] = []
    hist_score: List[float] = []
    hist_contour: List[float] = []
    hist_xnorm: List[float] = []
    hist_clamp: List[int] = []
    hist_move: List[float] = []

    best: Dict[str, Any] = {
        "score": np.inf,
        "centers": centers0.copy(),
        "fit": None,
    }

    n_eval = 0

    def objective(p: Array) -> float:
        nonlocal n_eval, best

        centers = _assemble_centers(p)
        key = _cache_key(centers)

        if key in cache:
            out = cache[key]
        else:
            # ---- NEW: hard reject if vessel avoidance is active and candidate violates ----
            if vessel_avoid.enabled:
                bad = vessel_violation_mask(
                    vessel_boundary=vessel_boundary,
                    centers=centers,
                    coil_a=coil_a,
                    clearance=float(vessel_avoid.clearance),
                    movable=movable,
                )
                if bool(np.any(bad)):
                    # huge score dominates; keep move term so DE prefers "less bad" moves
                    return float(1e12 + cfg.w_move * _move_m2(centers))

            A = build_A(centers)
            fit = solve_inner(A)

            contour = _contour_metric(fit)
            x = _x_vector(fit)
            xnorm2 = float(np.dot(x, x))
            clamp = _clamp_count(fit)
            move = _move_m2(centers)

            score = (
                cfg.w_contour * contour
                + cfg.w_xnorm * xnorm2
                + cfg.w_clamp * (clamp / max(1, Nc))
                + cfg.w_move * move
            )

            out = {
                "score": float(score),
                "contour": float(contour),
                "xnorm2": float(xnorm2),
                "clamp": int(clamp),
                "move": float(move),
                "fit": fit,
                "centers": centers,
                "p": np.asarray(p, float).copy(),
            }
            cache[key] = out

        # record history for every evaluation
        n_eval += 1
        hist_p.append(out["p"])
        hist_score.append(out["score"])
        hist_contour.append(out["contour"])
        hist_xnorm.append(out["xnorm2"])
        hist_clamp.append(out["clamp"])
        hist_move.append(out["move"])

        # update best
        if out["score"] < best["score"]:
            best = {"score": out["score"], "centers": out["centers"], "fit": out["fit"]}

        if cfg.verbose and (n_eval == 1 or (cfg.print_every and n_eval % cfg.print_every == 0)):
            print(
                f"[pos-opt] eval {n_eval:6d}: score={out['score']:.6g} "
                f"contour={out['contour']:.6g} clamps={out['clamp']} "
                f"xnorm2={out['xnorm2']:.6g} move_m2={out['move']:.6g} "
                f"best={best['score']:.6g}"
            )

        return out["score"]

    # ---- run differential evolution ----
    if not cfg.enabled or len(movable) == 0:
        # Degenerate case: no optimization, but return the same schema.
        fit0 = solve_inner(build_A(centers0))
        contour0 = _contour_metric(fit0)
        x0 = _x_vector(fit0)
        xnorm0 = float(np.dot(x0, x0))
        clamp0 = _clamp_count(fit0)
        move0 = 0.0
        score0 = (
            cfg.w_contour * contour0
            + cfg.w_xnorm * xnorm0
            + cfg.w_clamp * (clamp0 / max(1, Nc))
            + cfg.w_move * move0
        )
        return {
            "best_centers": centers0.copy(),
            "best_score": float(score0),
            "best_fit": fit0,
            "history": {
                "p": np.zeros((0, 2 * len(movable)), float),
                "score": np.zeros((0,), float),
                "contour_rms": np.zeros((0,), float),
                "x_norm": np.zeros((0,), float),
                "clamp_count": np.zeros((0,), int),
                "move_m2": np.zeros((0,), float),
            },
            "de_success": True,
            "de_message": "Optimization disabled or no movable coils",
            "de_nit": 0,
            "n_eval": 1,
            "vessel_avoidance": {
                "requested_enabled": bool((vessel_avoid or VesselAvoidanceConfig()).enabled),
                "applied_enabled": bool(vessel_avoid.enabled),
                "clearance": float(vessel_avoid.clearance),
                "initial_violations": list(initial_vessel_violations),
            },
        }

    def _callback(xk: Array, convergence: float) -> bool:
        # SciPy's "convergence" value can be uninformative when tol/atol are 0.
        # We print it for completeness, but the robust progress metric is best-score.
        if cfg.verbose:
            print(f"[pos-opt] gen-cb: conv={float(convergence):.6g} best={best['score']:.6g}")
        return False  # never early-stop from callback

    result = differential_evolution(
        objective,
        bounds_de,
        strategy="best1bin",
        maxiter=int(cfg.max_iter),
        popsize=int(cfg.popsize),
        tol=float(cfg.tol),
        atol=float(cfg.atol),
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=int(cfg.seed),
        polish=bool(cfg.polish),
        updating="deferred",
        workers=1,  # deterministic; parallel later if wanted
        callback=_callback,
        disp=False,
        init="latinhypercube",
    )

    # Ensure best is consistent with SciPy's final x if polishing changed it
    centers_fin = _assemble_centers(result.x)
    A_fin = build_A(centers_fin)
    fit_fin = solve_inner(A_fin)

    contour_fin = _contour_metric(fit_fin)
    x_fin = _x_vector(fit_fin)
    xnorm_fin = float(np.dot(x_fin, x_fin))
    clamp_fin = _clamp_count(fit_fin)
    move_fin = _move_m2(centers_fin)
    score_fin = (
        cfg.w_contour * contour_fin
        + cfg.w_xnorm * xnorm_fin
        + cfg.w_clamp * (clamp_fin / max(1, Nc))
        + cfg.w_move * move_fin
    )

    # Choose the true best between our tracked best and the final polished point.
    if float(score_fin) <= float(best["score"]):
        best_centers = centers_fin
        best_fit = fit_fin
        best_score = float(score_fin)
    else:
        best_centers = np.asarray(best["centers"], float)
        best_fit = best["fit"]
        best_score = float(best["score"])

    # Pack history arrays
    history = {
        "p": np.asarray(hist_p, float) if len(hist_p) else np.zeros((0, 2 * len(movable)), float),
        "score": np.asarray(hist_score, float) if len(hist_score) else np.zeros((0,), float),
        "contour_rms": np.asarray(hist_contour, float) if len(hist_contour) else np.zeros((0,), float),
        "x_norm": np.asarray(hist_xnorm, float) if len(hist_xnorm) else np.zeros((0,), float),
        "clamp_count": np.asarray(hist_clamp, int) if len(hist_clamp) else np.zeros((0,), int),
        "move_m2": np.asarray(hist_move, float) if len(hist_move) else np.zeros((0,), float),
    }

    return {
        "best_centers": best_centers,
        "best_score": best_score,
        "best_fit": best_fit,
        "history": history,
        "de_success": bool(result.success),
        "de_message": str(result.message),
        "de_nit": int(getattr(result, "nit", cfg.max_iter)),
        "n_eval": int(n_eval),
        "vessel_avoidance": {
            "requested_enabled": bool(vessel_avoid_applied or bool(vessel_avoid.enabled)),
            "applied_enabled": bool(vessel_avoid.enabled),
            "clearance": float(vessel_avoid.clearance),
            "initial_violations": list(initial_vessel_violations),
        },
    }
