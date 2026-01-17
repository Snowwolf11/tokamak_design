#!/usr/bin/env python3
"""
04_fit_pf_currents.py
=====================

Fit PF coil currents (and optionally PF coil *positions*) so that the target LCFS
is approximately a flux surface of the *vacuum* field produced by the PF coils.

This step is a "shape control" inverse problem. It intentionally uses only coils
(vacuum field) as actuators; plasma current is handled later in the free-boundary
equilibrium solve (05).

Two modes
---------
A) Current fit only (default):
   • Uses precomputed coil Green's functions on the full grid:
       /device/coil_greens/psi_per_amp  (Nc, NZ, NR)
   • Builds the boundary response matrix A from these greens at boundary points
   • Solves for currents (with regularization + optional bounds)

B) Current fit + coil position optimization (optional):
   • Outer loop changes coil centers (R,Z) for selected coils within bounds
   • Inner loop fits currents for each candidate using a boundary-only A
     computed analytically via tokdesign.physics.greens.psi_from_loop
     (fast: avoids recomputing full-grid greens each candidate)
   • After optimization, writes the best centers AND recomputes full-grid greens
     once (so downstream steps have consistent /device/coil_greens/psi_per_amp)

Inputs
------
From HDF5 (results.h5):
  /grid/R, /grid/Z, /grid/RR, /grid/ZZ
  /device/coil_greens/psi_per_amp        (Nc, NZ, NR)  [mode A]
  /device/coils/names                    (Nc,) optional
  /device/coils/centers                  (Nc,2)
  /device/coils/I_max                    (Nc,)
  /target/boundary                       (Nb,2)
  /target/psi_boundary                   scalar (target ψ value on LCFS; often 0)

From solver.yaml (coil_fit section):
  coil_fit:
    method: "boundary_value" | "contour" | "contour_qp"
    reg_lambda: float
    boundary_fit_points: int (optional; downsample boundary)
    weight_by_coil_limits: bool
    enforce_bounds: bool
    bounds_method: (metadata only; coil_fit handles bounds internally)

    # For contour methods:
    psi_ref: float         # desired mean (or other constraint) boundary ψ value
    constraint: "mean"     # currently supported by coil_fit contour solvers

    optimize_positions:
      enabled: bool
      movable_coils: "pf_only" | "all" | "none" | [list of names]
      R_bounds: [Rmin, Rmax]      # global fallback for movable coils
      Z_bounds: [Zmin, Zmax]
      per_coil_bounds:
        COILNAME:
          R: [Rmin, Rmax]
          Z: [Zmin, Zmax]
      max_iter: int
      population: int
      seed: int
      polish: bool
      # Optional override for the outer search inner-fit constraint:
      psi_ref: float
      constraint: str
      objective:
        w_contour: float
        w_xnorm: float
        w_clamp: float
        w_move: float

Outputs (results.h5)
-------------------
Updates (canonical "latest"):
  /device/coils/I_pf                    (Nc,)  fitted currents

If optimize_positions.enabled:
  /device/coils/centers                 (Nc,2) updated centers
  /device/coil_greens/psi_per_amp       (Nc, NZ, NR) recomputed for new centers

Writes:
  /optimization/fit_results/*
    I_pf_fit, boundary_points, psi_boundary_fit, residual, residual_rms,
    A, b, clamped, offset, psi_boundary_std, psi_boundary_ptp,
    plus contour extras if applicable (contour_rms, psi_ref, constraint)

If optimize_positions.enabled:
  /optimization/position_fit/*          (best + history of evaluated candidates)

History
-------
Anything overwritten is snapshotted to:
  /history/04_fit_pf_currents/<EVENT_ID>/...

Usage
-----
python scripts/04_fit_pf_currents.py \
  --run-dir data/runs/<RUN_ID> \
  --solver data/runs/<RUN_ID>/inputs/solver.yaml \
  [--overwrite] [--log-level INFO]

Notes
-----
• This script depends on:
    01_build_device.py (for coil greens + coil list)
    02_target_boundary.py (for target boundary)
• Step 04 is intended to be useful/diagnostic on its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.paths import assert_is_run_dir
from tokdesign.io.h5 import (
    open_h5,
    h5_ensure_group,
    h5_read_array,
    h5_read_scalar,
    h5_write_array,
    h5_write_scalar,
    h5_write_dict_as_attrs,
    h5_snapshot_paths,
    h5_make_history_event_id,
)
from tokdesign.io.schema import validate_h5_structure

from tokdesign.optimization.coil_fit import (
    fit_pf_currents_to_boundary,
    fit_pf_currents_to_boundary_from_A,
)
from tokdesign.optimization.coil_position_opt import (
    PositionOptConfig,
    optimize_coil_positions,
)
from tokdesign.physics.greens import psi_from_loop


# ============================================================
# HELPERS
# ============================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _as_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return default


def _decode_strings(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    arr = np.asarray(arr)
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _resample_boundary(boundary: np.ndarray, npts: int) -> np.ndarray:
    """
    Downsample a closed or open boundary polyline to ~npts points.

    This is a simple index-based downsample (not arc-length resampling).
    Good enough for v1 fits; can be upgraded later if needed.
    """
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("boundary must have shape (Nb,2)")
    if boundary.shape[0] < 4:
        raise ValueError("boundary too short to resample")

    # Remove closure duplicate if present
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]

    Nb = boundary.shape[0]
    if npts >= Nb:
        return boundary.copy()

    idx = np.linspace(0, Nb - 1, npts, endpoint=False)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, Nb - 1)
    idx = np.unique(idx)
    return boundary[idx, :]


def _build_A_boundary_from_loops(boundary_pts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Build A[k,c] = psi_per_amp at boundary point k from coil c,
    using analytic loop ψ at boundary points (fast).

    Parameters
    ----------
    boundary_pts : (Nb,2) array [m]
    centers      : (Nc,2) array [m]

    Returns
    -------
    A : (Nb,Nc)
    """
    boundary_pts = np.asarray(boundary_pts, float)
    centers = np.asarray(centers, float)
    if boundary_pts.ndim != 2 or boundary_pts.shape[1] != 2:
        raise ValueError("boundary_pts must be (Nb,2)")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must be (Nc,2)")

    Rb = boundary_pts[:, 0]
    Zb = boundary_pts[:, 1]
    Nc = centers.shape[0]
    Nb = boundary_pts.shape[0]

    A = np.empty((Nb, Nc), dtype=float)
    for c in range(Nc):
        Rc, Zc = float(centers[c, 0]), float(centers[c, 1])
        A[:, c] = np.asarray(psi_from_loop(Rb, Zb, Rc, Zc, I=1.0), float)
    return A


def _select_movable_indices(
    *,
    movable_spec: Union[str, Sequence[str]],
    coil_names: Optional[List[str]],
    Nc: int,
) -> List[int]:
    if isinstance(movable_spec, str):
        s = movable_spec.strip().lower()
        if s == "none":
            return []
        if s in ("all", "pf_only"):
            # v1: all coils in /device/coils are PF (including optional solenoid)
            return list(range(Nc))
        # otherwise treat as a single name string
        if coil_names is None:
            return []
        return [i for i, n in enumerate(coil_names) if n == movable_spec]
    # list of names
    if coil_names is None:
        return []
    name_to_idx = {n: i for i, n in enumerate(coil_names)}
    idx: List[int] = []
    for n in movable_spec:
        if n in name_to_idx:
            idx.append(name_to_idx[n])
    return sorted(set(idx))


def _bounds_for_movable(
    *,
    movable_idx: Sequence[int],
    coil_names: Optional[List[str]],
    centers: np.ndarray,
    pos_cfg: Dict[str, Any],
) -> List[Tuple[float, float, float, float]]:
    """
    Return per-movable coil bounds as (Rmin, Rmax, Zmin, Zmax).
    Uses global fallback bounds + optional per_coil_bounds overrides.
    """
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must be (Nc,2)")

    # global fallback
    Rb = pos_cfg.get("R_bounds", None)
    Zb = pos_cfg.get("Z_bounds", None)

    if Rb is None:
        Rmin_g, Rmax_g = float(np.min(centers[:, 0])), float(np.max(centers[:, 0]))
    else:
        Rmin_g, Rmax_g = float(Rb[0]), float(Rb[1])

    if Zb is None:
        Zmin_g, Zmax_g = float(np.min(centers[:, 1])), float(np.max(centers[:, 1]))
    else:
        Zmin_g, Zmax_g = float(Zb[0]), float(Zb[1])

    per = pos_cfg.get("per_coil_bounds", {}) or {}

    out: List[Tuple[float, float, float, float]] = []
    for i in movable_idx:
        nm = coil_names[i] if (coil_names is not None and i < len(coil_names)) else f"coil{i}"
        b_i = per.get(nm, {}) or {}

        Ri = b_i.get("R", [Rmin_g, Rmax_g])
        Zi = b_i.get("Z", [Zmin_g, Zmax_g])

        out.append((float(Ri[0]), float(Ri[1]), float(Zi[0]), float(Zi[1])))

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PF coil currents to target boundary flux (optional: optimize coil positions).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing results.h5")
    parser.add_argument("--solver", type=str, required=True, help="Path to solver.yaml (prefer archived copy in run_dir/inputs)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing /optimization/fit_results")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    assert_is_run_dir(run_dir)
    results_path = run_dir / "results.h5"
    if not results_path.exists():
        raise FileNotFoundError(f"results.h5 not found: {results_path}")

    logger = setup_logger(run_dir / "run.log", level=args.log_level)
    logger.info("Running 04_fit_pf_currents.py")
    logger.info("run_dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))

    solver_yaml = Path(args.solver).expanduser().resolve()
    if not solver_yaml.exists():
        raise FileNotFoundError(f"solver.yaml not found: {solver_yaml}")

    solver_cfg = _load_yaml(solver_yaml)
    coil_fit_cfg = (solver_cfg.get("coil_fit", {}) or {})

    # Core fit config
    method = str(coil_fit_cfg.get("method", "contour_qp")).strip()
    reg_lambda = float(coil_fit_cfg.get("reg_lambda", 1e-6))
    boundary_fit_points = coil_fit_cfg.get("boundary_fit_points", None)
    weight_by_limits = _as_bool(coil_fit_cfg.get("weight_by_coil_limits", True), default=True)
    enforce_bounds = _as_bool(coil_fit_cfg.get("enforce_bounds", True), default=True)
    bounds_method = str(coil_fit_cfg.get("bounds_method", "active_set"))  # metadata; coil_fit handles bounds internally

    # Contour settings
    psi_ref = float(coil_fit_cfg.get("psi_ref", 1.0))
    constraint = str(coil_fit_cfg.get("constraint", "mean"))

    # Optional position optimization config
    pos_cfg = (coil_fit_cfg.get("optimize_positions") or {})
    pos_enabled = _as_bool(pos_cfg.get("enabled", False), default=False)

    method_l = method.lower().strip()
    if method_l not in ("boundary_value", "ridge", "tikhonov", "contour", "flux_surface", "lcfs", "contour_qp", "qp"):
        raise ValueError(
            f"Unknown coil_fit.method='{method}'. "
            "Use 'boundary_value' or 'contour' or 'contour_qp'."
        )

    logger.info(
        "coil_fit config: method=%s, reg_lambda=%g, boundary_fit_points=%s, weight_by_limits=%s, enforce_bounds=%s, bounds_method=%s",
        method, reg_lambda, str(boundary_fit_points), str(weight_by_limits), str(enforce_bounds), bounds_method
    )
    if method_l in ("contour", "flux_surface", "lcfs", "contour_qp", "qp"):
        logger.info("contour settings: psi_ref=%g, constraint=%s", psi_ref, constraint)
    else:
        logger.info("boundary-value mode: will match psi_target on boundary (with optional offset fit)")

    # -----------------------------
    # Load data from HDF5
    # -----------------------------
    with open_h5(results_path, "r") as h5:
        schema_version = str(h5_read_scalar(h5, "/meta/schema_version"))
        run_id = str(h5_read_scalar(h5, "/meta/run_id"))

        # Grid
        R = np.asarray(h5_read_array(h5, "/grid/R"), float)
        Z = np.asarray(h5_read_array(h5, "/grid/Z"), float)

        RR = np.asarray(h5_read_array(h5, "/grid/RR"), float)
        ZZ = np.asarray(h5_read_array(h5, "/grid/ZZ"), float)

        # Coils
        centers0 = np.asarray(h5_read_array(h5, "/device/coils/centers"), float)  # (Nc,2)
        I_max = np.asarray(h5_read_array(h5, "/device/coils/I_max"), float).reshape(-1)

        coil_names = None
        if "/device/coils/names" in h5:
            coil_names = _decode_strings(h5_read_array(h5, "/device/coils/names"))

        # Precomputed full-grid greens (mode A; also used as baseline reference)
        G_psi = None
        if "/device/coil_greens/psi_per_amp" in h5:
            G_psi = np.asarray(h5_read_array(h5, "/device/coil_greens/psi_per_amp"), float)

        # Target
        boundary = np.asarray(h5_read_array(h5, "/target/boundary"), float)
        psi_target = float(h5_read_scalar(h5, "/target/psi_boundary"))

    if centers0.ndim != 2 or centers0.shape[1] != 2:
        raise ValueError("HDF5 /device/coils/centers must have shape (Nc,2).")
    Nc = centers0.shape[0]

    if I_max.size != Nc:
        raise ValueError(f"I_max size mismatch: expected Nc={Nc}, got {I_max.size}.")
    if np.any(~np.isfinite(I_max)) or np.any(I_max <= 0.0):
        raise ValueError("I_max must be finite and > 0 for all coils.")

    if boundary_fit_points is not None:
        boundary_fit = _resample_boundary(boundary, int(boundary_fit_points))
    else:
        boundary_fit = boundary[:-1] if (boundary.shape[0] >= 2 and np.allclose(boundary[0], boundary[-1])) else boundary

    Nb_fit = int(boundary_fit.shape[0])
    logger.info("Boundary points: original=%d, fit=%d", int(boundary.shape[0]), Nb_fit)
    logger.info("psi_target (/target/psi_boundary): %g", psi_target)

    # -----------------------------
    # Bounds for the inner current fit
    # -----------------------------
    # If weight_by_limits: solve in x = I/I_max with x bounds in [-1,1]
    if weight_by_limits:
        bounds_inner = (np.full(Nc, -1.0), np.full(Nc, +1.0)) if enforce_bounds else None
        logger.info("Using limit-weighted variables: x = I / I_max (bounds x in [-1,1])")
    else:
        bounds_inner = (-I_max, +I_max) if enforce_bounds else None
        logger.info("Using physical currents directly (bounds I in [-I_max, +I_max])")

    # -----------------------------
    # Optional: position optimization outer loop
    # -----------------------------
    centers_final = centers0.copy()
    position_result: Optional[Dict[str, object]] = None

    if pos_enabled:
        logger.info("Position optimization enabled.")

        movable_spec = pos_cfg.get("movable_coils", "pf_only")
        movable_idx = _select_movable_indices(movable_spec=movable_spec, coil_names=coil_names, Nc=Nc)
        logger.info("Movable coils: %d / %d", len(movable_idx), Nc)

        if len(movable_idx) == 0:
            logger.warning("Position optimization enabled but no movable coils selected. Proceeding with current fit only.")
        else:
            bounds_RZ = _bounds_for_movable(
                movable_idx=movable_idx,
                coil_names=coil_names,
                centers=centers0,
                pos_cfg=pos_cfg,
            )

            # Outer-loop inner constraint can be overridden (often useful)
            psi_ref_outer = float(pos_cfg.get("psi_ref", psi_ref))
            constraint_outer = str(pos_cfg.get("constraint", constraint))

            obj_cfg = (pos_cfg.get("objective") or {})
            opt_cfg = PositionOptConfig(
                enabled=True,
                max_iter=int(pos_cfg.get("max_iter", 80)),
                popsize=int(pos_cfg.get("population", 18)),
                seed=int(pos_cfg.get("seed", 0)),
                polish=_as_bool(pos_cfg.get("polish", True), default=True),
                tol=float(pos_cfg.get("tol", 0.0) or 0.0),
                atol=float(pos_cfg.get("atol", 0.0) or 0.0),
                w_contour=float(obj_cfg.get("w_contour", 1.0)),
                w_xnorm=float(obj_cfg.get("w_xnorm", 0.02)),
                w_clamp=float(obj_cfg.get("w_clamp", 2.0)),
                w_move=float(obj_cfg.get("w_move", 0.1)),
                cache_round_m=float(pos_cfg.get("cache_round_m", 1e-3)),
            )

            # Build A at boundary points analytically for candidate centers (fast)
            def build_A_for_centers(centers_try: np.ndarray) -> np.ndarray:
                A = _build_A_boundary_from_loops(boundary_fit, centers_try)
                if weight_by_limits:
                    # scaled columns for x = I/I_max
                    return A * I_max[None, :]
                return A

            # Inner solve operating on A directly (keeps outer loop light)
            def solve_inner(A: np.ndarray) -> Dict[str, np.ndarray]:
                # For contour methods, psi_target is not meaningful (it is usually 0),
                # so use psi_ref_outer and constraint_outer.
                return fit_pf_currents_to_boundary_from_A(
                    A=A,
                    psi_target=psi_target,
                    reg_lambda=reg_lambda,
                    I_bounds=bounds_inner if enforce_bounds else None,
                    method=method,
                    fit_offset=True,
                    psi_ref=psi_ref_outer,
                    constraint=constraint_outer,
                )

            logger.info(
                "Position opt settings: max_iter=%d pop=%d seed=%d polish=%s  (weights: contour=%g xnorm=%g clamp=%g move=%g)",
                opt_cfg.max_iter, opt_cfg.popsize, opt_cfg.seed, str(opt_cfg.polish),
                opt_cfg.w_contour, opt_cfg.w_xnorm, opt_cfg.w_clamp, opt_cfg.w_move
            )

            position_result = optimize_coil_positions(
                centers0=centers0,
                movable=movable_idx,
                bounds_RZ=bounds_RZ,
                build_A=build_A_for_centers,
                solve_inner=solve_inner,
                cfg=opt_cfg,
                I_max=I_max if weight_by_limits else None,
            )
            centers_final = np.asarray(position_result["best_centers"], float)
            logger.info(
                "Position opt done: best_score=%g n_eval=%d success=%s nit=%s",
                float(position_result["best_score"]),
                int(position_result["n_eval"]),
                str(position_result["de_success"]),
                str(position_result["de_nit"]),
            )
            if not bool(position_result["de_success"]):
                logger.warning("Position optimizer reported failure: %s", str(position_result["de_message"]))

    # -----------------------------
    # Final current fit (always): use final centers / best geometry
    # -----------------------------
    logger.info("Fitting PF currents to boundary (final)...")

    if pos_enabled and position_result is not None:
        # Use analytic boundary A (consistent with geometry) and do the inner solve
        A_final = _build_A_boundary_from_loops(boundary_fit, centers_final)
        if weight_by_limits:
            A_final = A_final * I_max[None, :]

        fit = fit_pf_currents_to_boundary_from_A(
            A=A_final,
            psi_target=psi_target,
            reg_lambda=reg_lambda,
            I_bounds=bounds_inner if enforce_bounds else None,
            method=method,
            fit_offset=True,
            psi_ref=psi_ref,
            constraint=constraint,
        )
    else:
        # Use precomputed full-grid greens path (mode A)
        if G_psi is None:
            raise FileNotFoundError(
                "Missing /device/coil_greens/psi_per_amp in results.h5, required for current-fit-only mode.\n"
                "Re-run 01_build_device.py with precompute_coil_greens enabled, or enable optimize_positions."
            )
        # Scale greens for x variables if desired
        G_use = (G_psi * I_max[:, None, None]) if weight_by_limits else G_psi
        fit = fit_pf_currents_to_boundary(
            G_psi=G_use,
            boundary_pts=boundary_fit,
            R=R,
            Z=Z,
            psi_target=psi_target,
            reg_lambda=reg_lambda,
            I_bounds=bounds_inner if enforce_bounds else None,
            method=method,
            fit_offset=True,
            psi_ref=psi_ref,
            constraint=constraint,
        )

    # -----------------------------
    # Interpret inner-fit outputs
    # -----------------------------
    # The fit dict uses "I_fit" as the solve variable. If weight_by_limits=True,
    # that variable is x = I/I_max. Convert to physical currents for storage.
    I_var = np.asarray(fit["I_fit"], float).reshape(-1)
    if I_var.size != Nc:
        raise RuntimeError(f"coil_fit returned wrong size I_fit: {I_var.size} != Nc={Nc}")

    if weight_by_limits:
        x_fit = I_var.copy()
        I_fit = x_fit * I_max
        store_x = True
    else:
        I_fit = I_var.copy()
        store_x = False

    psi_boundary_fit = np.asarray(fit["psi_boundary_fit"], float).reshape(-1)
    residual = np.asarray(fit["residual"], float).reshape(-1)
    residual_rms = float(np.asarray(fit["residual_rms"]).item())
    clamped = np.asarray(fit.get("clamped", np.zeros(Nc, dtype=bool)), bool).reshape(-1)

    offset = float(np.asarray(fit.get("offset", 0.0)).item())
    psi_std = float(np.asarray(fit.get("psi_boundary_std", np.std(psi_boundary_fit))).item())
    psi_ptp = float(np.asarray(fit.get("psi_boundary_ptp", np.ptp(psi_boundary_fit))).item())

    is_contour = method_l in ("contour", "flux_surface", "lcfs", "contour_qp", "qp")
    if is_contour:
        contour_rms = float(np.asarray(fit.get("contour_rms", residual_rms)).item())
        logger.info("Contour RMS (RMS of dpsi along boundary): %g (Wb/rad)", contour_rms)
        logger.info("Boundary psi std: %g  ptp: %g  mean: %g (Wb/rad)", psi_std, psi_ptp, float(np.mean(psi_boundary_fit)))
        logger.info("psi_ref=%g constraint=%s", psi_ref, constraint)
    else:
        logger.info("Boundary-value RMS: %g (Wb/rad), offset=%g (Wb/rad)", residual_rms, offset)
        logger.info("Boundary psi std: %g  ptp: %g  mean: %g (Wb/rad)", psi_std, psi_ptp, float(np.mean(psi_boundary_fit)))

    logger.info("Clamp count: %d / %d coils", int(np.sum(clamped)), Nc)
    logger.info("I_fit range: [%g, %g] A", float(np.min(I_fit)), float(np.max(I_fit)))

    # -----------------------------
    # Write outputs to HDF5 (with history snapshot)
    # -----------------------------
    with open_h5(results_path, "r+") as h5:
        # overwrite guard for fit_results
        if not args.overwrite and "/optimization/fit_results" in h5:
            raise RuntimeError(
                "It looks like /optimization/fit_results already exists.\n"
                "Refusing to overwrite. Re-run with --overwrite."
            )

        # Snapshot anything we will overwrite
        event_id = h5_make_history_event_id()
        snap_paths: List[str] = ["/device/coils/I_pf"]

        if pos_enabled and position_result is not None:
            snap_paths.append("/device/coils/centers")
            if "/device/coil_greens/psi_per_amp" in h5:
                snap_paths.append("/device/coil_greens/psi_per_amp")

        if "/optimization/fit_results" in h5:
            snap_paths.append("/optimization/fit_results")
        if pos_enabled and "/optimization/position_fit" in h5:
            snap_paths.append("/optimization/position_fit")

        h5_snapshot_paths(
            h5,
            stage="04_fit_pf_currents",
            event_id=event_id,
            src_paths=snap_paths,
            attrs={
                "script": "04_fit_pf_currents.py",
                "method": str(method),
                "reg_lambda": float(reg_lambda),
                "Nb_fit": int(Nb_fit),
                "position_opt": bool(pos_enabled and position_result is not None),
                "note": "snapshot before writing new PF fit results",
            },
            overwrite_event=False,
        )

        # Ensure groups
        h5_ensure_group(h5, "/device/coils")
        h5_ensure_group(h5, "/optimization")
        h5_ensure_group(h5, "/optimization/fit_results")

        # If positions were optimized, apply new centers and recompute full-grid greens once
        if pos_enabled and position_result is not None:
            h5_write_array(h5, "/device/coils/centers", centers_final, attrs={"units": "m"}, overwrite=True)

            # Recompute full-grid greens for final centers to keep /device/coil_greens consistent
            Nc_check = centers_final.shape[0]
            if Nc_check != Nc:
                raise RuntimeError("Internal error: centers_final Nc mismatch")

            G_new = np.empty((Nc, RR.shape[0], RR.shape[1]), dtype=float)
            for c in range(Nc):
                Rc, Zc = float(centers_final[c, 0]), float(centers_final[c, 1])
                G_new[c] = np.asarray(psi_from_loop(RR, ZZ, Rc, Zc, I=1.0), float)

            h5_ensure_group(h5, "/device/coil_greens")
            h5_write_array(
                h5,
                "/device/coil_greens/psi_per_amp",
                G_new,
                attrs={"units": "Wb_per_rad_per_A", "method": "analytic_loop (psi_from_loop)"},
                overwrite=True,
            )

        # Update canonical currents
        h5_write_array(h5, "/device/coils/I_pf", I_fit, attrs={"units": "A"}, overwrite=True)

        # Fit results
        h5_write_array(h5, "/optimization/fit_results/I_pf_fit", I_fit, attrs={"units": "A"}, overwrite=True)
        if store_x:
            h5_write_array(h5, "/optimization/fit_results/x_fit", x_fit, attrs={"units": "I/I_max"}, overwrite=True)

        h5_write_array(h5, "/optimization/fit_results/boundary_points", boundary_fit, attrs={"units": "m"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/psi_boundary_fit", psi_boundary_fit, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/residual", residual, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/residual_rms", residual_rms, attrs={"units": "Wb/rad"}, overwrite=True)

        # Store A and b (very helpful for debugging)
        if "A" in fit:
            h5_write_array(
                h5, "/optimization/fit_results/A", np.asarray(fit["A"], float),
                attrs={"meaning": "A[k,c]=psi_per_amp at boundary point k from coil c (scaled if weight_by_limits)"},
                overwrite=True
            )
        if "b" in fit:
            h5_write_array(
                h5, "/optimization/fit_results/b", np.asarray(fit["b"], float),
                attrs={"meaning": "reference b used by fit (method-dependent)"},
                overwrite=True
            )

        h5_write_array(
            h5, "/optimization/fit_results/clamped", clamped.astype(np.uint8),
            attrs={"meaning": "1 if coil ended at a bound"}, overwrite=True
        )

        # Common diagnostics
        h5_write_scalar(h5, "/optimization/fit_results/offset", offset, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/psi_boundary_std", psi_std, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/psi_boundary_ptp", psi_ptp, attrs={"units": "Wb/rad"}, overwrite=True)

        # Contour extras
        if is_contour:
            contour_rms = float(np.asarray(fit.get("contour_rms", residual_rms)).item())
            h5_write_scalar(h5, "/optimization/fit_results/contour_rms", contour_rms, attrs={"units": "Wb/rad"}, overwrite=True)
            h5_write_scalar(h5, "/optimization/fit_results/psi_ref", float(psi_ref), attrs={"units": "Wb/rad"}, overwrite=True)
            h5_write_dict_as_attrs(h5, "/optimization/fit_results", {"constraint": str(constraint)}, overwrite=True)

        # Metadata attrs
        h5_write_dict_as_attrs(h5, "/optimization/fit_results", {
            "method": str(method),
            "reg_lambda": float(reg_lambda),
            "weight_by_coil_limits": bool(weight_by_limits),
            "enforce_bounds": bool(enforce_bounds),
            "bounds_method": str(bounds_method),
            "psi_target": float(psi_target),   # provenance even if contour uses psi_ref
            "Nb_fit": int(Nb_fit),
            "Nc": int(Nc),
            "position_opt_enabled": bool(pos_enabled),
            "position_opt_applied": bool(pos_enabled and position_result is not None),
        }, overwrite=True)

        # Position optimization outputs (optional)
        if pos_enabled and position_result is not None:
            h5_ensure_group(h5, "/optimization/position_fit")
            h5_ensure_group(h5, "/optimization/position_fit/best")
            h5_ensure_group(h5, "/optimization/position_fit/history")
            h5_ensure_group(h5, "/optimization/objective_terms")
            h5_ensure_group(h5, "/optimization/constraint_margins")

            h5_write_array(h5, "/optimization/position_fit/best/centers", centers_final, attrs={"units": "m"}, overwrite=True)
            h5_write_scalar(h5, "/optimization/position_fit/best/score", float(position_result["best_score"]), overwrite=True)

            hist = position_result["history"]
            h5_write_array(h5, "/optimization/position_fit/history/p", np.asarray(hist["p"], float), overwrite=True)
            h5_write_array(h5, "/optimization/position_fit/history/score", np.asarray(hist["score"], float), overwrite=True)
            h5_write_array(h5, "/optimization/position_fit/history/contour_rms", np.asarray(hist["contour_rms"], float), overwrite=True)
            h5_write_array(h5, "/optimization/position_fit/history/x_norm", np.asarray(hist["x_norm"], float), overwrite=True)
            h5_write_array(h5, "/optimization/position_fit/history/clamp_count", np.asarray(hist["clamp_count"], int), overwrite=True)
            h5_write_array(h5, "/optimization/position_fit/history/move_m2", np.asarray(hist["move_m2"], float), overwrite=True)

            h5_write_dict_as_attrs(h5, "/optimization/position_fit", {
                "n_eval": int(position_result["n_eval"]),
                "de_success": bool(position_result["de_success"]),
                "de_nit": int(position_result["de_nit"]),
                "de_message": str(position_result["de_message"]),
                "movable_spec": str(pos_cfg.get("movable_coils", "pf_only")),
            }, overwrite=True)
            
            # objective_terms (always safe to store)
            h5_write_scalar(h5, "/optimization/objective_terms/reg_lambda", float(reg_lambda), overwrite=True)

            if is_contour:
                h5_write_scalar(h5, "/optimization/objective_terms/contour_rms",
                                float(np.asarray(fit.get("contour_rms", residual_rms)).item()),
                                attrs={"units": "Wb/rad"}, overwrite=True)

            # a useful "effort" metric
            if weight_by_limits and store_x:
                h5_write_scalar(h5, "/optimization/objective_terms/x_norm2",
                                float(np.dot(x_fit, x_fit)), overwrite=True)
            else:
                # physical currents (dimensioned; still useful as a norm)
                h5_write_scalar(h5, "/optimization/objective_terms/I_norm2",
                                float(np.dot(I_fit, I_fit)), attrs={"units": "A^2"}, overwrite=True)

            h5_write_scalar(h5, "/optimization/objective_terms/clamp_count",
                            int(np.sum(clamped)), overwrite=True)
            
            if enforce_bounds:
                if weight_by_limits and store_x:
                    # x in [-1,1]
                    margin = 1.0 - np.abs(x_fit)
                    h5_write_array(h5, "/optimization/constraint_margins/margin_to_x_bounds",
                                margin, overwrite=True)
                    h5_write_scalar(h5, "/optimization/constraint_margins/min_margin_to_x_bounds",
                                    float(np.min(margin)), overwrite=True)
                else:
                    # I in [-I_max, I_max]
                    margin = I_max - np.abs(I_fit)
                    h5_write_array(h5, "/optimization/constraint_margins/margin_to_I_bounds",
                                margin, attrs={"units": "A"}, overwrite=True)
                    h5_write_scalar(h5, "/optimization/constraint_margins/min_margin_to_I_bounds",
                                    float(np.min(margin)), attrs={"units": "A"}, overwrite=True)
    
    logger.info("Wrote fitted currents to /device/coils/I_pf and results to /optimization/fit_results/*")

    # Validate structure (stage="optimization" if your schema supports it; otherwise "complete")
    try:
        validate_h5_structure(results_path, schema_version=schema_version, stage="optimization_fit")
        logger.info("Schema validation passed for stage='optimization_fit'")
    except Exception as e:
        # Don't hard-fail; schema may not yet have this stage defined.
        logger.warning("Schema validation for stage='optimizatio_fit' failed or not defined: %s", str(e))

    print("\nPF coil fit complete.")
    print(f"  run_id:       {run_id}")
    print(f"  run_dir:      {run_dir}")
    print(f"  results.h5:   {results_path}")
    print(f"  Nc:           {Nc}")
    print(f"  Nb_fit:       {Nb_fit}")
    if is_contour:
        contour_rms = float(np.asarray(fit.get('contour_rms', residual_rms)).item())
        print(f"  contour_rms:  {contour_rms:g} (Wb/rad)")
    else:
        print(f"  residual_rms: {residual_rms:g} (Wb/rad)")
    if pos_enabled and position_result is not None:
        print("  position_opt: yes (applied)")
    else:
        print("  position_opt: no")
    print("")


if __name__ == "__main__":
    main()