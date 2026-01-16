#!/usr/bin/env python3
"""
04_fit_pf_currents.py
=====================

Fit PF coil currents to reproduce a *target boundary flux* on the target LCFS.

Goal
----
Solve a (regularized) linear inverse problem for coil currents I such that the
*vacuum* poloidal flux from the PF coils matches a desired constant value
psi_target on a set of boundary points:

    psi_vac(Rk,Zk) = sum_c G_psi[c](Rk,Zk) * I[c]  ≈  psi_target

This is a standard "shape control" / boundary fitting step. It does NOT include
plasma current. It produces coil currents that place the desired ψ contour near
the target boundary, providing:
  • a machine realization of the target shape
  • a good starting point for the later free-boundary equilibrium solve (05)

Inputs
------
From HDF5 (results.h5):
  /grid/R, /grid/Z
  /device/coil_greens/psi_per_amp        (Nc, NZ, NR)
  /device/coils/I_max                   (Nc,)
  /target/boundary                      (Nb,2)
  /target/psi_boundary                  scalar (target ψ on LCFS)

From solver.yaml (coil_fit section):
  coil_fit:
    reg_lambda: float
    method: "ridge" (v1; placeholder)
    boundary_fit_points: int (optional; downsample boundary)
    weight_by_coil_limits: bool (optional; recommended)
    enforce_bounds: bool (optional)
    bounds_method: "clip" (v1; placeholder)

Outputs (results.h5)
-------------------
Updates:
  /device/coils/I_pf                    (Nc,)  fitted currents

Writes:
  /optimization/fit_results/*
    I_pf_fit            (Nc,)
    psi_boundary_fit    (Nb_fit,)
    residual            (Nb_fit,)
    A                   (Nb_fit, Nc)   (optional but very useful for debugging)
    b                   (Nb_fit,)
    clamped             (Nc,) bool
    residual_rms        scalar
    (plus metadata attrs)

Usage
-----
python scripts/04_fit_pf_currents.py \
  --run-dir data/runs/<RUN_ID> \
  --solver data/runs/<RUN_ID>/inputs/solver.yaml \
  [--overwrite] [--log-level INFO]

Notes
-----
• This script depends on:
    01_build_device.py (for coil greens)
    02_target_boundary.py (for target boundary)
• It intentionally uses ONLY coils (vacuum field) as the actuators.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
import h5py

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

from tokdesign.optimization.coil_fit import fit_pf_currents_to_boundary


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


def _resample_boundary(boundary: np.ndarray, npts: int) -> np.ndarray:
    """
    Resample boundary polyline to npts points (not including a closure duplicate),
    then return an array of shape (npts,2).
    """
    boundary = np.asarray(boundary, dtype=float)
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("boundary must have shape (Nb,2)")

    if boundary.shape[0] < 4:
        raise ValueError("boundary too short to resample")

    # Remove duplicate closure point if present (common convention: first==last)
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]

    Nb = boundary.shape[0]
    if npts >= Nb:
        return boundary.copy()

    idx = np.linspace(0, Nb - 1, npts, endpoint=False)
    idx = np.round(idx).astype(int)
    idx = np.clip(idx, 0, Nb - 1)
    # Ensure unique indices (in case Nb small)
    idx = np.unique(idx)
    return boundary[idx, :]


def _get_schema_version(h5: h5py.File) -> str:
    sv = h5_read_scalar(h5, "/meta/schema_version")
    return str(sv)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit PF coil currents to target boundary flux.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing results.h5")
    parser.add_argument("--solver", type=str, required=True,
                        help="Path to solver.yaml (prefer archived copy in run_dir/inputs)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fit results and I_pf")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    assert_is_run_dir(run_dir)
    results_path = run_dir / "results.h5"
    if not results_path.exists():
        raise FileNotFoundError(f"results.h5 not found in run directory: {results_path}")

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path, level=args.log_level)
    logger.info("Running 04_fit_pf_currents.py")
    logger.info("run_dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))

    solver_yaml = Path(args.solver).expanduser().resolve()
    if not solver_yaml.exists():
        raise FileNotFoundError(f"solver.yaml not found: {solver_yaml}")

    solver_cfg = _load_yaml(solver_yaml)
    coil_fit_cfg = (solver_cfg.get("coil_fit", {}) or {})

    method = str(coil_fit_cfg.get("method", "ridge"))
    reg_lambda = float(coil_fit_cfg.get("reg_lambda", 1e-6))
    boundary_fit_points = coil_fit_cfg.get("boundary_fit_points", None)
    weight_by_limits = _as_bool(coil_fit_cfg.get("weight_by_coil_limits", False), default=False)
    enforce_bounds = _as_bool(coil_fit_cfg.get("enforce_bounds", True), default=True)
    bounds_method = str(coil_fit_cfg.get("bounds_method", "clip"))
    psi_ref = float(coil_fit_cfg.get("psi_ref", 1.0))
    constraint = str(coil_fit_cfg.get("constraint", "mean"))

    if method.lower() not in ("ridge", "contour"):
        logger.warning("Unknown coil_fit.method='%s' (v1 uses ridge or contour). Proceeding anyway.", method)

    if reg_lambda < 0:
        raise ValueError("coil_fit.reg_lambda must be >= 0")

    if boundary_fit_points is not None:
        boundary_fit_points = int(boundary_fit_points)
        if boundary_fit_points < 16:
            raise ValueError("coil_fit.boundary_fit_points should be >= 16 for a stable boundary fit")

    logger.info("coil_fit config: method=%s, reg_lambda=%g, boundary_fit_points=%s, weight_by_limits=%s, "
                "enforce_bounds=%s, bounds_method=%s",
                method, reg_lambda, str(boundary_fit_points), str(weight_by_limits),
                str(enforce_bounds), bounds_method)

    # -----------------------------
    # Read inputs from HDF5
    # -----------------------------
    with open_h5(results_path, "r") as h5:
        schema_version = _get_schema_version(h5)

    # Validate required stages exist
    validate_h5_structure(results_path, schema_version=schema_version, stage="device")
    validate_h5_structure(results_path, schema_version=schema_version, stage="target")

    with open_h5(results_path, "r") as h5:
        R = h5_read_array(h5, "/grid/R")
        Z = h5_read_array(h5, "/grid/Z")

        G_psi = h5_read_array(h5, "/device/coil_greens/psi_per_amp")  # (Nc, NZ, NR)
        I_max = h5_read_array(h5, "/device/coils/I_max")              # (Nc,)

        boundary = h5_read_array(h5, "/target/boundary")              # (Nb,2)
        psi_target = float(h5_read_scalar(h5, "/target/psi_boundary"))

    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    G_psi = np.asarray(G_psi, float)
    I_max = np.asarray(I_max, float).reshape(-1)
    boundary = np.asarray(boundary, float)

    Nc, NZ, NR = G_psi.shape
    if I_max.size != Nc:
        raise ValueError(f"I_max size mismatch: expected Nc={Nc}, got {I_max.size}")

    if np.any(I_max <= 0) or not np.all(np.isfinite(I_max)):
        raise ValueError("Invalid I_max values: must be finite and > 0 for all coils")

    if boundary_fit_points is not None:
        boundary_fit = _resample_boundary(boundary, boundary_fit_points)
    else:
        # also remove closure duplicate if present
        boundary_fit = boundary[:-1] if (boundary.shape[0] >= 2 and np.allclose(boundary[0], boundary[-1])) else boundary

    Nb_fit = boundary_fit.shape[0]
    logger.info("Boundary points: original=%d, fit=%d", boundary.shape[0], Nb_fit)
    logger.info("psi_target (LCFS): %g", psi_target)

    # -----------------------------
    # Weighting by coil limits (recommended option)
    # -----------------------------
    # If weight_by_limits:
    #   solve in scaled variables x = I / I_max
    #   psi = sum_c (G_c * I_max[c]) * x_c
    # Then:
    #   minimize ||A_scaled x - b||^2 + λ||x||^2
    # Bounds become x in [-1,1] if enforce_bounds.
    if weight_by_limits:
        logger.info("Using limit-weighted variables: x = I / I_max")
        G_use = G_psi * I_max[:, None, None]
        # bounds in x
        bounds = (np.full(Nc, -1.0), np.full(Nc, +1.0)) if enforce_bounds else None
    else:
        G_use = G_psi
        bounds = (-I_max, +I_max) if enforce_bounds else None

    if enforce_bounds and bounds_method.lower() not in ("clip", "clip_thrn_refit",  "active_set", "box"):
        logger.warning("Unknown bounds_method='%s'. Using built-in clamp/active-set in coil_fit.", bounds_method)

    # -----------------------------
    # Fit currents
    # -----------------------------
    logger.info("Fitting PF currents (vacuum) to boundary flux...")
    fit = fit_pf_currents_to_boundary(
        G_psi=G_use,
        boundary_pts=boundary_fit,
        R=R,
        Z=Z,
        psi_target=psi_target,
        reg_lambda=reg_lambda,
        I_bounds=bounds,
        method=method,                 # <<< NEW
        fit_offset=True,               # used only in boundary_value mode
        psi_ref=psi_ref,               # used only in contour mode
        constraint=constraint,         # used only in contour mode
    )





    A = np.asarray(fit["A"], float)
    b = np.asarray(fit["b"], float)
    I = np.asarray(fit["I_fit"], float)
    offset = float(np.asarray(fit.get("offset", 0.0)).item())

    print("DEBUG A stats:",
        "shape", A.shape,
        "min", np.min(A), "max", np.max(A),
        "col_norms", np.linalg.norm(A, axis=0))

    print("DEBUG b stats:",
        "min", np.min(b), "max", np.max(b),
        "std", np.std(b))

    print("DEBUG solution:",
        "||I||", np.linalg.norm(I),
      "offset", offset)
    psi_fit = np.asarray(fit["psi_boundary_fit"], float)
    print("DEBUG psi_fit:", "min", psi_fit.min(), "max", psi_fit.max(), "std", psi_fit.std())
    Rmin, Rmax = float(R[0]), float(R[-1])
    Zmin, Zmax = float(Z[0]), float(Z[-1])
    Rb = boundary_fit[:,0]; Zb = boundary_fit[:,1]
    print("DEBUG grid R:", Rmin, Rmax, "Z:", Zmin, Zmax)
    print("DEBUG boundary R:", float(Rb.min()), float(Rb.max()), "Z:", float(Zb.min()), float(Zb.max()))





    I_fit = np.asarray(fit["I_fit"], float).reshape(-1)
    if I_fit.size != Nc:
        raise RuntimeError("coil_fit returned wrong I_fit size")

    # Unscale back to physical currents if using x = I/Imax
    if weight_by_limits:
        x_fit = I_fit
        I_phys = x_fit * I_max
        # For reporting, overwrite returned I_fit to physical
        I_fit = I_phys

        # Also re-compute boundary fit ψ from physical currents:
        # psi = sum_c G_psi[c] * I[c] (use original G_psi)
        # We already have A (built from G_use), but that corresponds to G*Imax.
        # The psi fit from the solver is still correct (A_scaled @ x), so we can keep it.
        # We'll store both x and I for clarity.
        store_x = True
    else:
        store_x = False

    
    psi_boundary_fit = np.asarray(fit["psi_boundary_fit"], float)
    mean_psi = float(np.mean(psi_boundary_fit))
    residual = np.asarray(fit["residual"], float)
    residual_rms = float(np.asarray(fit["residual_rms"]).item())
    clamped = np.asarray(fit["clamped"], bool)

    offset = float(np.asarray(fit.get("offset", 0.0)).item())
    psi_std = float(np.asarray(fit["psi_boundary_std"]).item())
    psi_ptp = float(np.asarray(fit["psi_boundary_ptp"]).item())

    logger.info("Boundary psi std: %g (Wb/rad)", psi_std)
    logger.info("Boundary psi ptp: %g (Wb/rad)", psi_ptp)

    logger.info("Mean boundary psi: %g (target psi_ref=%g)", mean_psi, psi_ref)

    is_contour = str(method).lower().strip() in ("contour", "flux_surface", "lcfs")
    if is_contour:
        contour_rms = float(np.asarray(fit.get("contour_rms", fit["residual_rms"])).item())
        logger.info("Contour RMS (RMS of dpsi along boundary): %g (Wb/rad)", contour_rms)
        logger.info("psi_ref=%g constraint=%s", psi_ref, constraint)
        logger.info("Fit done: contour_rms=%g (Wb/rad)", contour_rms)
    else:
        logger.info("Fit offset c: %g (Wb/rad)", offset)
        logger.info("Boundary value RMS: %g (Wb/rad)", residual_rms)
        logger.info("Fit done: residual_rms=%g (Wb/rad)", residual_rms)

    logger.info("Clamp count: %d / %d coils", int(np.sum(clamped)), Nc)
    logger.info("I_fit range: [%g, %g] A", float(np.min(I_fit)), float(np.max(I_fit)))


    # -----------------------------
    # Write outputs to HDF5
    # -----------------------------
    with open_h5(results_path, "r+") as h5:
        if not args.overwrite:
            if "/optimization/fit_results" in h5 or "/optimization/fit_results/I_pf_fit" in h5:
                raise RuntimeError(
                    "It looks like /optimization/fit_results already exists.\n"
                    "Refusing to overwrite. Re-run with --overwrite."
                )

        h5_ensure_group(h5, "/optimization")
        h5_ensure_group(h5, "/optimization/fit_results")
        h5_ensure_group(h5, "/device/coils")

        # ------------------------------------------------------------
        # History snapshot (before overwriting /device/coils/I_pf and fit_results)
        # ------------------------------------------------------------
        event_id = h5_make_history_event_id()

        h5_snapshot_paths(
            h5,
            stage="04_fit_pf_currents",
            event_id=event_id,
            src_paths=[
                "/device/coils/I_pf",
                "/optimization/fit_results",  # snapshot previous fit results too, if any
            ],
            attrs={
                "script": "04_fit_pf_currents.py",
                "method": str(method),
                "reg_lambda": float(reg_lambda),
                "Nb_fit": int(Nb_fit),
                "note": "snapshot before writing new coil fit results",
            },
            overwrite_event=False,
        )
        # Update device coil currents (canonical latest)
        h5_write_array(h5, "/device/coils/I_pf", I_fit, attrs={"units": "A"}, overwrite=True)

        # Fit results
        h5_write_array(h5, "/optimization/fit_results/I_pf_fit", I_fit, attrs={"units": "A"}, overwrite=True)
        if store_x:
            h5_write_array(
                h5, "/optimization/fit_results/x_fit", np.asarray(fit["I_fit"], float),
                attrs={"units": "I/I_max"}, overwrite=True
            )

        h5_write_array(h5, "/optimization/fit_results/boundary_points", boundary_fit, attrs={"units": "m"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/psi_boundary_fit", psi_boundary_fit,
                    attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/residual", residual, attrs={"units": "Wb/rad"}, overwrite=True)

        # Keep residual_rms for backward compatibility; interpret depends on method
        h5_write_scalar(h5, "/optimization/fit_results/residual_rms", residual_rms,
                        attrs={"units": "Wb/rad"}, overwrite=True)

        # Store A, b for debugging (optional but very useful)
        h5_write_array(h5, "/optimization/fit_results/A", np.asarray(fit["A"], float),
                    attrs={"meaning": "A[k,c]=psi_per_amp at boundary point k from coil c"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/b", np.asarray(fit["b"], float),
                    attrs={"meaning": "reference b used by fit (method-dependent)"}, overwrite=True)
        h5_write_array(h5, "/optimization/fit_results/clamped", clamped.astype(np.uint8),
                    attrs={"meaning": "1 if coil ended at a bound"}, overwrite=True)

        # Common diagnostics
        h5_write_scalar(h5, "/optimization/fit_results/offset", offset, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/psi_boundary_std", psi_std, attrs={"units": "Wb/rad"}, overwrite=True)
        h5_write_scalar(h5, "/optimization/fit_results/psi_boundary_ptp", psi_ptp, attrs={"units": "Wb/rad"}, overwrite=True)

        # Contour-specific extras
        if is_contour:
            contour_rms = float(np.asarray(fit.get("contour_rms", fit["residual_rms"])).item())
            h5_write_scalar(h5, "/optimization/fit_results/contour_rms", contour_rms,
                            attrs={"units": "Wb/rad"}, overwrite=True)
            h5_write_scalar(h5, "/optimization/fit_results/psi_ref", float(psi_ref),
                            attrs={"units": "Wb/rad"}, overwrite=True)
            h5_write_dict_as_attrs(h5, "/optimization/fit_results", {"constraint": constraint}, overwrite=True)

        # Metadata attrs
        h5_write_dict_as_attrs(h5, "/optimization/fit_results", {
            "method": str(method),
            "reg_lambda": float(reg_lambda),
            "weight_by_coil_limits": bool(weight_by_limits),
            "enforce_bounds": bool(enforce_bounds),
            "bounds_method": str(bounds_method),
            "psi_target": float(psi_target),   # kept for provenance even if contour ignores it
            "Nb_fit": int(Nb_fit),
            "Nc": int(Nc),
        }, overwrite=True)

    logger.info("Wrote fitted currents to /device/coils/I_pf and results to /optimization/fit_results/*")

    print("\nPF coil fit complete.")
    print(f"  run_dir:      {run_dir}")
    print(f"  results.h5:   {results_path}")
    print(f"  Nc:           {Nc}")
    print(f"  Nb_fit:       {Nb_fit}")
    print(f"  residual_rms: {residual_rms:g} (Wb/rad)")
    print("")


if __name__ == "__main__":
    main()