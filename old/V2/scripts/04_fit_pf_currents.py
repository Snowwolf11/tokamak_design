#!/usr/bin/env python3
"""
04_fit_pf_currents.py
=====================

Stage 04 orchestrator: fit PF coil currents to match the target LCFS boundary
(and optionally optimize PF coil positions), then write results to results.h5.

This version:
  • Keeps the "new" algorithmic fixes (psi_ref handling, xnorm handling,
    schema-required outputs).
  • Restores the "old" write-out behavior (overwrite canonical coil data but
    snapshot previous values into history first).
  • Adds optional vessel-avoidance for position optimization:
      - Reads vessel boundary from /device/vessel_boundary
      - Reads coil radii a from /device/coils/radii
      - Passes data to coil_position_opt (no HDF5 access inside src optimizers)
      - If starting positions violate and vessel_avoidance.enabled=True, stage 04
        logs a warning (coil_position_opt returns the violating indices and disables
        the constraint internally for this run).

CLI
---
python scripts/04_fit_pf_currents.py \
  --run-dir data/runs/<RUN_ID> \
  --solver data/runs/<RUN_ID>/inputs/solver.yaml \
  [--overwrite] [--log-level INFO]
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
from tokdesign.optimization.vessel_avoidance import VesselAvoidanceConfig
from tokdesign.physics._greens import psi_from_loop


# ============================================================
# small helpers (orchestrator-level)
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
    for x in np.asarray(arr):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _resample_boundary(boundary: np.ndarray, npts: int) -> np.ndarray:
    """
    Cheap downsample of boundary points to reduce computation cost.
    (Index-based, not arc-length reparameterization.)
    """
    boundary = np.asarray(boundary, float)
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("boundary must have shape (Nb,2)")

    # Drop duplicate closing point if present
    if boundary.shape[0] >= 2 and np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]

    Nb = boundary.shape[0]
    if npts is None or npts <= 0 or npts >= Nb:
        return boundary.copy()

    idx = np.linspace(0, Nb - 1, npts, endpoint=False)
    idx = np.unique(np.clip(np.round(idx).astype(int), 0, Nb - 1))
    return boundary[idx]


def _build_A_boundary_from_loops(boundary_pts: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Boundary-only response matrix:
      A[k,c] = psi_per_amp at boundary point k from coil c
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


def _rebuild_full_grid_greens_from_loops(R: np.ndarray, Z: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Recompute full-grid psi_per_amp for each coil using filament loop psi.

    Returns:
      G_psi: (Nc, NZ, NR)
      with meshgrid built as RR,ZZ = np.meshgrid(R, Z)  -> (NZ, NR)
    """
    R = np.asarray(R, float).reshape(-1)
    Z = np.asarray(Z, float).reshape(-1)
    centers = np.asarray(centers, float)

    RR, ZZ = np.meshgrid(R, Z)  # shapes (NZ, NR)
    Nc = centers.shape[0]

    G = np.empty((Nc, ZZ.shape[0], RR.shape[1]), dtype=float)
    for c in range(Nc):
        Rc, Zc = float(centers[c, 0]), float(centers[c, 1])
        G[c, :, :] = np.asarray(psi_from_loop(RR, ZZ, Rc, Zc, I=1.0), float)
    return G


def _select_movable_indices(
    *,
    movable_spec: Union[str, Sequence[str]],
    coil_names: Optional[List[str]],
    Nc: int,
) -> List[int]:
    """
    Map a user-facing spec to coil indices.
    v1 behavior:
      - "none"    => []
      - "all"     => [0..Nc-1]
      - "pf_only" => [0..Nc-1]  (until coil metadata distinguishes PF/TF/etc)
      - list/str of names => matched by /device/coils/names if present
    """
    if isinstance(movable_spec, str):
        s = movable_spec.strip().lower()
        if s == "none":
            return []
        if s in ("all", "pf_only"):
            return list(range(Nc))
        # treat as single name
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
    Return per-movable coil bounds:
      (Rmin, Rmax, Zmin, Zmax)

    Supports:
      - global pos_cfg["R_bounds"] / ["Z_bounds"]
      - optional per-coil overrides in pos_cfg["per_coil_bounds"]
    """
    centers = np.asarray(centers, float)

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

    per = (pos_cfg.get("per_coil_bounds", {}) or {})

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
    parser = argparse.ArgumentParser(description="Stage 04: Fit PF currents (optional: PF coil position optimization).")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing results.h5")
    parser.add_argument("--solver", type=str, required=True, help="Path to solver.yaml")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing optimization results")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    assert_is_run_dir(run_dir)
    results_path = run_dir / "results.h5"
    if not results_path.exists():
        raise FileNotFoundError(f"results.h5 not found: {results_path}")

    solver_yaml = Path(args.solver).expanduser().resolve()
    if not solver_yaml.exists():
        raise FileNotFoundError(f"solver.yaml not found: {solver_yaml}")

    logger = setup_logger(run_dir / "run.log", level=args.log_level)
    logger.info("Running stage 04 (PF current fit)")
    logger.info("run_dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))
    logger.info("solver.yaml: %s", str(solver_yaml))

    solver_cfg = _load_yaml(solver_yaml)
    coil_fit_cfg = (solver_cfg.get("coil_fit", {}) or {})

    # ---- core config ----
    method = str(coil_fit_cfg.get("method", "contour_qp")).strip()
    reg_lambda = float(coil_fit_cfg.get("reg_lambda", 1e-6))
    boundary_fit_points = coil_fit_cfg.get("boundary_fit_points", None)

    weight_by_limits = _as_bool(coil_fit_cfg.get("weight_by_coil_limits", True), default=True)
    enforce_bounds = _as_bool(coil_fit_cfg.get("enforce_bounds", True), default=True)
    bounds_method = str(coil_fit_cfg.get("bounds_method", "active_set"))  # metadata only

    # ---- optional position optimization ----
    pos_cfg = (coil_fit_cfg.get("optimize_positions") or {})
    pos_enabled = _as_bool(pos_cfg.get("enabled", False), default=False)

    # ---- optional vessel avoidance (only used if pos_enabled) ----
    va_cfg = (pos_cfg.get("vessel_avoidance") or {})
    vessel_avoid = VesselAvoidanceConfig(
        enabled=_as_bool(va_cfg.get("enabled", False), default=False),
        clearance=float(va_cfg.get("clearance", 0.0) or 0.0),
    )

    method_l = method.lower().strip()
    is_contour = method_l in ("contour", "flux_surface", "lcfs", "contour_qp", "qp")

    # -----------------------------
    # Load data from HDF5
    # -----------------------------
    with open_h5(results_path, "r") as h5:
        schema_version = str(h5_read_scalar(h5, "/meta/schema_version"))
        run_id = str(h5_read_scalar(h5, "/meta/run_id"))

        # Grid
        R = np.asarray(h5_read_array(h5, "/grid/R"), float)
        Z = np.asarray(h5_read_array(h5, "/grid/Z"), float)

        # Coils
        centers0 = np.asarray(h5_read_array(h5, "/device/coils/centers"), float)  # (Nc,2)
        I_max = np.asarray(h5_read_array(h5, "/device/coils/I_max"), float).reshape(-1)

        coil_names = None
        if "/device/coils/names" in h5:
            coil_names = _decode_strings(h5_read_array(h5, "/device/coils/names"))

        # Full-grid greens (may be required if we do current-fit-only mode)
        G_psi = None
        if "/device/coil_greens/psi_per_amp" in h5:
            G_psi = np.asarray(h5_read_array(h5, "/device/coil_greens/psi_per_amp"), float)

        # Target boundary and its known psi value
        boundary = np.asarray(h5_read_array(h5, "/target/boundary"), float)
        psi_target = float(h5_read_scalar(h5, "/target/psi_boundary"))

        # NEW: vessel boundary + coil radii (for vessel avoidance)
        vessel_boundary = None
        coil_radii = None
        if "/device/vessel_boundary" in h5:
            vessel_boundary = np.asarray(h5_read_array(h5, "/device/vessel_boundary"), float)
        if "/device/coils/radii" in h5:
            coil_radii = np.asarray(h5_read_array(h5, "/device/coils/radii"), float)

    if centers0.ndim != 2 or centers0.shape[1] != 2:
        raise ValueError("HDF5 /device/coils/centers must have shape (Nc,2)")
    Nc = centers0.shape[0]

    if I_max.size != Nc:
        raise ValueError(f"I_max size mismatch: expected Nc={Nc}, got {I_max.size}")
    if np.any(~np.isfinite(I_max)) or np.any(I_max <= 0.0):
        raise ValueError("I_max must be finite and > 0 for all coils")

    # If vessel avoidance is requested, ensure data is present
    if pos_enabled and vessel_avoid.enabled:
        if vessel_boundary is None:
            raise FileNotFoundError("vessel_avoidance.enabled=True but missing /device/vessel_boundary in results.h5")
        if coil_radii is None:
            raise FileNotFoundError("vessel_avoidance.enabled=True but missing /device/coils/radii in results.h5")

    # Boundary points used by fit/optimizer
    Nb_raw = int(boundary.shape[0])
    if boundary_fit_points is not None:
        boundary_fit = _resample_boundary(boundary, int(boundary_fit_points))
    else:
        boundary_fit = _resample_boundary(boundary, 0)
    Nb_fit = int(boundary_fit.shape[0])

    logger.info("Boundary points: raw=%d fit=%d", Nb_raw, Nb_fit)
    logger.info("psi_target (/target/psi_boundary): %g", psi_target)

    if pos_enabled and vessel_avoid.enabled:
        logger.info(
            "Vessel avoidance requested: enabled=True clearance=%g (RZ vessel boundary + coil radii loaded)",
            float(vessel_avoid.clearance),
        )

    # -----------------------------
    # Choose psi_ref and constraint for contour methods
    # -----------------------------
    constraint = str(coil_fit_cfg.get("constraint", "mean"))
    psi_ref_cfg = coil_fit_cfg.get("psi_ref", None)

    if is_contour:
        # Fix: default psi_ref is the *known* LCFS psi from the target
        psi_ref = float(psi_target if psi_ref_cfg is None else psi_ref_cfg)
        logger.info(
            "Contour method: using psi_ref=%g (default=psi_target unless overridden), constraint=%s",
            psi_ref, constraint
        )
    else:
        psi_ref = float(psi_ref_cfg) if psi_ref_cfg is not None else float(psi_target)
        logger.info(
            "Boundary-value method: match psi_target=%g (psi_ref=%g stored for provenance)",
            psi_target, psi_ref
        )

    # -----------------------------
    # Bounds for inner current fit
    # -----------------------------
    if weight_by_limits:
        # Variable is x = I/I_max
        bounds_inner = (np.full(Nc, -1.0), np.full(Nc, +1.0)) if enforce_bounds else None
        logger.info("Current variables: x = I/I_max (bounds x in [-1,1])")
    else:
        # Variable is I in amps
        bounds_inner = (-I_max, +I_max) if enforce_bounds else None
        logger.info("Current variables: I [A] (bounds I in [-I_max, +I_max])")

    # -----------------------------
    # Optional position optimization (outer loop)
    # -----------------------------
    centers_best = centers0.copy()
    pos_result: Optional[Dict[str, Any]] = None

    if pos_enabled:
        movable_spec = pos_cfg.get("movable_coils", "pf_only")
        movable_idx = _select_movable_indices(movable_spec=movable_spec, coil_names=coil_names, Nc=Nc)
        logger.info("Position optimization enabled: movable coils %d/%d", len(movable_idx), Nc)

        if len(movable_idx) > 0:
            bounds_RZ = _bounds_for_movable(
                movable_idx=movable_idx,
                coil_names=coil_names,
                centers=centers0,
                pos_cfg=pos_cfg,
            )

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

            # Allow (optional) overrides for outer loop, but default to correct values.
            psi_ref_outer = float(pos_cfg.get("psi_ref", psi_ref))
            constraint_outer = str(pos_cfg.get("constraint", constraint))

            # build A for candidate centers
            def build_A_for_centers(centers_try: np.ndarray) -> np.ndarray:
                A = _build_A_boundary_from_loops(boundary_fit, centers_try)
                return (A * I_max[None, :]) if weight_by_limits else A

            # inner solve used by the outer loop
            def solve_inner(A: np.ndarray) -> Dict[str, np.ndarray]:
                fit = fit_pf_currents_to_boundary_from_A(
                    A=A,
                    psi_target=psi_target,
                    reg_lambda=reg_lambda,
                    I_bounds=bounds_inner if enforce_bounds else None,
                    method=method,
                    fit_offset=True,
                    psi_ref=psi_ref_outer,
                    constraint=constraint_outer,
                )
                # Fix: when weight_by_limits=True, variable is x, so explicitly store x_fit.
                if weight_by_limits:
                    fit["x_fit"] = np.asarray(fit["I_fit"], float).copy()
                return fit

            # IMPORTANT FIX:
            # If weight_by_limits=True, the outer optimizer must not divide by I_max again.
            try:
                pos_result = optimize_coil_positions(
                    centers0=centers0,
                    movable=movable_idx,
                    bounds_RZ=bounds_RZ,
                    build_A=build_A_for_centers,
                    solve_inner=solve_inner,
                    cfg=opt_cfg,
                    var_mode=("x" if weight_by_limits else "I"),
                    I_max=(None if weight_by_limits else I_max),
                    # NEW: vessel avoidance pass-through
                    vessel_avoid=vessel_avoid,
                    vessel_boundary=vessel_boundary,
                    coil_a=coil_radii,
                )
            except TypeError:
                # Backward-compatible call (older API)
                pos_result = optimize_coil_positions(
                    centers0=centers0,
                    movable=movable_idx,
                    bounds_RZ=bounds_RZ,
                    build_A=build_A_for_centers,
                    solve_inner=solve_inner,
                    cfg=opt_cfg,
                    I_max=(None if weight_by_limits else I_max),
                )

            # If vessel avoidance was requested but disabled due to initial violations,
            # coil_position_opt reports the violating indices here.
            va_info = (pos_result or {}).get("vessel_avoidance", {}) if pos_result is not None else {}
            bad0 = va_info.get("initial_violations", []) if isinstance(va_info, dict) else []
            applied = bool(va_info.get("applied_enabled", False)) if isinstance(va_info, dict) else False
            if vessel_avoid.enabled and bad0:
                logger.warning(
                    "vessel_avoidance.enabled=True but initial movable coils intersect vessel for indices %s. "
                    "Constraint was disabled for this run.",
                    bad0,
                )
            if vessel_avoid.enabled and not applied and not bad0:
                logger.info("vessel_avoidance requested but not applied (optimizer disabled it or not supported).")
            if vessel_avoid.enabled and applied:
                logger.info("vessel_avoidance applied with clearance=%g", float(vessel_avoid.clearance))

            centers_best = np.asarray(pos_result["best_centers"], float)
            logger.info(
                "Position opt done: best_score=%g n_eval=%d success=%s nit=%s",
                float(pos_result["best_score"]),
                int(pos_result["n_eval"]),
                str(pos_result.get("de_success", True)),
                str(pos_result.get("de_nit", "?")),
            )
            if not bool(pos_result.get("de_success", True)):
                logger.warning("Position optimizer reported failure: %s", str(pos_result.get("de_message", "")))
        else:
            logger.warning("Position optimization enabled but movable set is empty; skipping outer loop.")

    # -----------------------------
    # Final current fit (always)
    # -----------------------------
    logger.info("Final current fit...")

    if pos_enabled and pos_result is not None:
        # Boundary-only A for best centers
        A_final = _build_A_boundary_from_loops(boundary_fit, centers_best)
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
        # Current-fit-only path: requires precomputed full-grid greens
        if G_psi is None:
            raise FileNotFoundError(
                "Missing /device/coil_greens/psi_per_amp in results.h5.\n"
                "Re-run 01_build_device.py with coil greens enabled, or enable optimize_positions."
            )
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
    # Convert outputs to canonical I_fit [A]
    # -----------------------------
    I_var = np.asarray(fit["I_fit"], float).reshape(-1)
    if I_var.size != Nc:
        raise RuntimeError(f"coil_fit returned I_fit of size {I_var.size}, expected Nc={Nc}")

    if weight_by_limits:
        x_fit = I_var.copy()           # solver variable is x
        I_fit = x_fit * I_max          # canonical amps
        fit["x_fit"] = x_fit
    else:
        x_fit = None
        I_fit = I_var.copy()

    psi_boundary_fit = np.asarray(fit["psi_boundary_fit"], float).reshape(-1)
    residual = np.asarray(fit["residual"], float).reshape(-1)
    residual_rms = float(np.asarray(fit["residual_rms"]).item())
    clamped = np.asarray(fit.get("clamped", np.zeros(Nc, dtype=bool)), bool).reshape(-1)

    offset = float(np.asarray(fit.get("offset", 0.0)).item())
    psi_std = float(np.asarray(fit.get("psi_boundary_std", np.std(psi_boundary_fit))).item())
    psi_ptp = float(np.asarray(fit.get("psi_boundary_ptp", np.ptp(psi_boundary_fit))).item())

    contour_rms = float(np.asarray(fit.get("contour_rms", residual_rms)).item()) if is_contour else None

    logger.info("Clamp count: %d/%d", int(np.sum(clamped)), Nc)
    logger.info("I_fit range [A]: [%g, %g]", float(np.min(I_fit)), float(np.max(I_fit)))
    if weight_by_limits and x_fit is not None:
        logger.info("x_fit range [-]: [%g, %g]", float(np.min(x_fit)), float(np.max(x_fit)))
        logger.info("x_norm2 [-]: %g", float(np.dot(x_fit, x_fit)))
    if is_contour:
        logger.info("contour_rms [Wb/rad]: %g  (psi_ref=%g constraint=%s)", float(contour_rms), psi_ref, constraint)
    else:
        logger.info("residual_rms [Wb/rad]: %g  offset=%g", residual_rms, offset)

    # -----------------------------
    # Write outputs (history snapshot + overwrite canonical device paths)
    # -----------------------------
    with open_h5(results_path, "r+") as h5:
        # overwrite guard
        if not args.overwrite:
            if "/optimization/fit_results" in h5 or "/optimization/position_fit" in h5:
                raise RuntimeError(
                    "Optimization outputs already exist. Re-run with --overwrite if you want to replace them."
                )

        # snapshot overwritten paths (OLD-behavior compatible)
        event_id = h5_make_history_event_id()
        snap_paths: List[str] = []

        # Canonical device state that may be overwritten in this stage
        if "/device/coils/centers" in h5:
            snap_paths.append("/device/coils/centers")
        if "/device/coil_greens/psi_per_amp" in h5:
            snap_paths.append("/device/coil_greens/psi_per_amp")
        if "/device/coils/I_pf" in h5:
            snap_paths.append("/device/coils/I_pf")

        # Optimization outputs we overwrite
        if "/optimization/fit_results" in h5:
            snap_paths.append("/optimization/fit_results")
        if "/optimization/position_fit" in h5:
            snap_paths.append("/optimization/position_fit")
        if "/optimization/objective_terms" in h5:
            snap_paths.append("/optimization/objective_terms")
        if "/optimization/constraint_margins" in h5:
            snap_paths.append("/optimization/constraint_margins")

        if snap_paths:
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
                    "position_opt_enabled": bool(pos_enabled),
                    "position_opt_applied": bool(pos_enabled and pos_result is not None),
                    "vessel_avoidance_requested": bool(pos_enabled and vessel_avoid.enabled),
                    "vessel_avoidance_clearance": float(vessel_avoid.clearance),
                    "note": "snapshot before overwriting stage-04 canonical device + optimization outputs",
                },
                overwrite_event=False,
            )

        # Ensure groups (also satisfies schema validator)
        h5_ensure_group(h5, "/device")
        h5_ensure_group(h5, "/device/coils")
        h5_ensure_group(h5, "/device/coil_greens")
        h5_ensure_group(h5, "/optimization")
        h5_ensure_group(h5, "/optimization/fit_results")
        h5_ensure_group(h5, "/optimization/objective_terms")
        h5_ensure_group(h5, "/optimization/constraint_margins")

        # ---- OLD-style: apply best centers and rebuild grid greens (if pos opt applied) ----
        if pos_enabled and pos_result is not None:
            # overwrite canonical centers
            h5_write_array(h5, "/device/coils/centers", centers_best, attrs={"units": "m"}, overwrite=True)

            # rebuild full-grid greens for updated centers (keeps quicklook consistent)
            G_new = _rebuild_full_grid_greens_from_loops(R=R, Z=Z, centers=centers_best)
            h5_write_array(
                h5,
                "/device/coil_greens/psi_per_amp",
                G_new,
                attrs={"meaning": "psi per amp (filament loop model), rebuilt after stage-04 center update"},
                overwrite=True,
            )

        # canonical fitted currents used by later steps
        h5_write_array(h5, "/device/coils/I_pf", I_fit, attrs={"units": "A"}, overwrite=True)

        # ---- store fit results (debug-friendly) ----
        h5_write_array(h5, "/optimization/fit_results/I_pf_fit", I_fit, attrs={"units": "A"}, overwrite=True)
        if x_fit is not None:
            h5_write_array(h5, "/optimization/fit_results/x_fit", x_fit, attrs={"units": "I/I_max"}, overwrite=True)

        h5_write_array(h5, "/optimization/fit_results/boundary_points", boundary_fit, attrs={"units": "m"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/psi_boundary_fit", psi_boundary_fit, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/residual", residual, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/residual_rms", residual_rms, attrs={"units": "Wb/rad"}, overwrite=True)

        # Optional internals (if inner solver returned them)
        if "A" in fit:
            h5_write_array(
                h5, "/optimization/fit_results/A", np.asarray(fit["A"], float),
                attrs={"meaning": "Boundary influence matrix (scaled if weight_by_limits=True)"},
                overwrite=True,
            )
        if "b" in fit:
            h5_write_array(
                h5, "/optimization/fit_results/b", np.asarray(fit["b"], float),
                attrs={"meaning": "Reference vector used by fit (method-dependent)"},
                overwrite=True,
            )

        h5_write_array(
            h5, "/optimization/fit_results/clamped", clamped.astype(np.uint8),
            attrs={"meaning": "1 if coil ended at a bound"},
            overwrite=True,
        )

        h5_write_scalar(h5, "/optimization/fit_results/offset", offset, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/psi_boundary_std", psi_std, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/psi_boundary_ptp", psi_ptp, attrs={"units": "Wb/rad"}, overwrite=True)

        if is_contour and contour_rms is not None:
            h5_write_scalar(h5, "/optimization/fit_results/contour_rms", float(contour_rms), attrs={"units": "Wb/rad"}, overwrite=True)
            h5_write_scalar(h5, "/optimization/fit_results/psi_ref", float(psi_ref), attrs={"units": "Wb/rad"}, overwrite=True)
            h5_write_dict_as_attrs(h5, "/optimization/fit_results", {"constraint": str(constraint)}, overwrite=True)

        # provenance attrs
        h5_write_dict_as_attrs(
            h5,
            "/optimization/fit_results",
            {
                "method": str(method),
                "reg_lambda": float(reg_lambda),
                "weight_by_coil_limits": bool(weight_by_limits),
                "enforce_bounds": bool(enforce_bounds),
                "bounds_method": str(bounds_method),
                "psi_target": float(psi_target),
                "psi_ref_used": float(psi_ref),
                "Nb_fit": int(Nb_fit),
                "Nc": int(Nc),
                "position_opt_enabled": bool(pos_enabled),
                "position_opt_applied": bool(pos_enabled and pos_result is not None),
                "vessel_avoidance_requested": bool(pos_enabled and vessel_avoid.enabled),
                "vessel_avoidance_clearance": float(vessel_avoid.clearance),
                "note": (
                    "Stage 04 overwrote canonical /device paths (centers, greens, I_pf) if position optimization ran; "
                    "previous values were saved via history snapshot."
                ),
            },
            overwrite=True,
        )

        # ---- schema-required diagnostics ----
        h5_write_scalar(h5, "/optimization/objective_terms/reg_lambda", float(reg_lambda), overwrite=True)
        h5_write_scalar(h5, "/optimization/objective_terms/clamp_count", int(np.sum(clamped)), overwrite=True)

        if is_contour and contour_rms is not None:
            h5_write_scalar(h5, "/optimization/objective_terms/contour_rms", float(contour_rms), attrs={"units": "Wb/rad"}, overwrite=True)

        if x_fit is not None:
            xnorm2 = float(np.dot(x_fit, x_fit))
            h5_write_scalar(h5, "/optimization/objective_terms/x_norm2", xnorm2, overwrite=True)
            if enforce_bounds:
                margin = 1.0 - np.abs(x_fit)
                h5_write_array(h5, "/optimization/constraint_margins/margin_to_x_bounds", margin, overwrite=True)
                h5_write_scalar(h5, "/optimization/constraint_margins/min_margin_to_x_bounds", float(np.min(margin)), overwrite=True)
        else:
            Inorm2 = float(np.dot(I_fit, I_fit))
            h5_write_scalar(h5, "/optimization/objective_terms/I_norm2", Inorm2, attrs={"units": "A^2"}, overwrite=True)
            if enforce_bounds:
                margin = I_max - np.abs(I_fit)
                h5_write_array(h5, "/optimization/constraint_margins/margin_to_I_bounds", margin, attrs={"units": "A"}, overwrite=True)
                h5_write_scalar(h5, "/optimization/constraint_margins/min_margin_to_I_bounds", float(np.min(margin)), attrs={"units": "A"}, overwrite=True)

        # ---- position optimization outputs (optional) ----
        if pos_enabled and pos_result is not None:
            h5_ensure_group(h5, "/optimization/position_fit")
            h5_ensure_group(h5, "/optimization/position_fit/best")
            h5_ensure_group(h5, "/optimization/position_fit/history")

            h5_write_array(h5, "/optimization/position_fit/best/centers", centers_best, attrs={"units": "m"}, overwrite=True)
            h5_write_scalar(h5, "/optimization/position_fit/best/score", float(pos_result["best_score"]), overwrite=True)

            hist = pos_result.get("history", {})
            if hist:
                h5_write_array(h5, "/optimization/position_fit/history/p", np.asarray(hist.get("p", []), float), overwrite=True)
                h5_write_array(h5, "/optimization/position_fit/history/score", np.asarray(hist.get("score", []), float), overwrite=True)
                h5_write_array(h5, "/optimization/position_fit/history/contour_rms", np.asarray(hist.get("contour_rms", []), float), overwrite=True)
                h5_write_array(h5, "/optimization/position_fit/history/x_norm", np.asarray(hist.get("x_norm", []), float), overwrite=True)
                h5_write_array(h5, "/optimization/position_fit/history/clamp_count", np.asarray(hist.get("clamp_count", []), int), overwrite=True)
                h5_write_array(h5, "/optimization/position_fit/history/move_m2", np.asarray(hist.get("move_m2", []), float), overwrite=True)

            # Include vessel-avoidance diagnostics from the optimizer (if present)
            va_info = pos_result.get("vessel_avoidance", {}) if isinstance(pos_result, dict) else {}
            h5_write_dict_as_attrs(
                h5,
                "/optimization/position_fit",
                {
                    "n_eval": int(pos_result.get("n_eval", 0)),
                    "de_success": bool(pos_result.get("de_success", True)),
                    "de_nit": int(pos_result.get("de_nit", 0)),
                    "de_message": str(pos_result.get("de_message", "")),
                    "movable_spec": str(pos_cfg.get("movable_coils", "pf_only")),
                    "psi_ref_used": float(psi_ref),
                    "constraint_used": str(constraint),
                    "vessel_avoidance_requested": bool(vessel_avoid.enabled),
                    "vessel_avoidance_clearance": float(vessel_avoid.clearance),
                    "vessel_avoidance_applied": bool(va_info.get("applied_enabled", False)) if isinstance(va_info, dict) else False,
                    "vessel_avoidance_initial_violations": str(va_info.get("initial_violations", [])) if isinstance(va_info, dict) else "[]",
                },
                overwrite=True,
            )

    logger.info("Stage 04 complete: wrote canonical /device/coils/I_pf and optimization results.")

    # Schema validation
    try:
        validate_h5_structure(results_path, schema_version=schema_version, stage="optimization_fit")
        logger.info("Schema validation passed for stage='optimization_fit'")
    except Exception as e:
        logger.warning("Schema validation for stage='optimization_fit' failed or not defined: %s", str(e))

    print("\nPF coil fit complete.")
    print(f"  run_id:       {run_id}")
    print(f"  run_dir:      {run_dir}")
    print(f"  results.h5:   {results_path}")
    print(f"  Nc:           {Nc}")
    print(f"  Nb_fit:       {Nb_fit}")
    if is_contour and contour_rms is not None:
        print(f"  contour_rms:  {contour_rms:g} (Wb/rad)")
    else:
        print(f"  residual_rms: {residual_rms:g} (Wb/rad)")
    print(f"  position_opt: {'yes' if (pos_enabled and pos_result is not None) else 'no'}")
    if pos_enabled and pos_result is not None:
        print("  note: canonical /device/coils/centers and /device/coil_greens/psi_per_amp were overwritten (old state snapshotted).")
    if pos_enabled and vessel_avoid.enabled:
        print(f"  vessel_avoidance: requested (clearance={vessel_avoid.clearance:g} m)")
    print("")


if __name__ == "__main__":
    main()
