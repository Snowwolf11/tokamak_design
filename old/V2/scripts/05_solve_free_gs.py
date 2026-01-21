#!/usr/bin/env python3
"""
scripts/05_solve_free_gs.py
===========================

Stage 05 runner: solve free-boundary Grad–Shafranov equilibrium.

Thin orchestration only:
- parse CLI
- load YAML configs
- read required inputs from results.h5
- call tokdesign.physics.solve_free_gs.solve_free_boundary_equilibrium
- write outputs back into results.h5

Assumptions (per your workflow):
- --device, --target, --solver are always provided
- configs are already archived by stage 00 (so no copying here)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import h5py

from tokdesign.io.h5 import (
    h5_read_array,
    h5_read_scalar,
    h5_write_array,
    h5_write_scalar,
    h5_ensure_group,
)
from tokdesign.io.logging_utils import setup_logger

from tokdesign.physics._solve_free_gs import (
    solve_free_boundary_equilibrium,
    FreeBoundaryConfig,
)


# -----------------------------------------------------------------------------
# YAML helpers
# -----------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_profile_from_target_cfg(target_cfg: dict[str, Any]) -> Any:
    """
    Build the profile parameters object expected by your GS profile functions.

    target_equilibrium.yaml layout:

      profiles:
        pressure:
          p0
          alpha_p
        toroidal_field_function:
          F0
          alpha_F
    """
    prof = target_cfg.get("profiles") or {}

    pres = prof.get("pressure") or {}
    tff = prof.get("toroidal_field_function") or {}

    p0 = float(pres.get("p0", 0.0))
    alpha_p = float(pres.get("alpha_p", 1.0))

    F0 = float(tff.get("F0", 0.0))
    alpha_F = float(tff.get("alpha_F", 0.0))

    try:
        from tokdesign.physics._gs_profiles import GSProfileParams
        return GSProfileParams(p0=p0, alpha_p=alpha_p, F0=F0, alpha_F=alpha_F)
    except Exception:
        return {"p0": p0, "alpha_p": alpha_p, "F0": F0, "alpha_F": alpha_F}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 05: solve free-boundary GS equilibrium")
    ap.add_argument("--run", "--run-dir", dest="run_dir", required=True,
                    help="Run directory containing results.h5")
    ap.add_argument("--device", required=True, type=str, help="Path to baseline_device.yaml")
    ap.add_argument("--target", required=True, type=str, help="Path to target_equilibrium.yaml")
    ap.add_argument("--solver", required=True, type=str, help="Path to solver.yaml")

    ap.add_argument("--use-fixed-init", action="store_true",
                    help="If /equilibrium/fixed/psi exists (from stage 03), use it as psi_plasma_init.")
    ap.add_argument("--no-fields", action="store_true",
                    help="Skip computing fields/derived; only write equilibrium + history.")
    ap.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...).")

    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    h5_path = run_dir / "results.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"results.h5 not found: {h5_path}")

    logger = setup_logger(str(run_dir / "stage_05_solve_free_gs.log"), level=args.log_level)
    logger.info("Stage 05 starting")
    logger.info("run_dir=%s", run_dir)
    logger.info("device=%s", args.device)
    logger.info("target=%s", args.target)
    logger.info("solver=%s", args.solver)

    # Load YAML configs
    _ = _load_yaml(Path(args.device).expanduser().resolve())  # currently unused; kept for symmetry/provenance
    target_cfg = _load_yaml(Path(args.target).expanduser().resolve())
    solver_cfg = _load_yaml(Path(args.solver).expanduser().resolve())

    # Build profile params from target_equilibrium.yaml
    profile = _build_profile_from_target_cfg(target_cfg)

    # -----------------------------------------------------------------------------
    # Free-boundary config handling (robust + transparent)
    # -----------------------------------------------------------------------------
    # Solver sub-configs
    fixed_cfg = solver_cfg.get("fixed_boundary_gs") or {}
    free_cfg_dict = solver_cfg.get("free_boundary") or {}

    if isinstance(free_cfg_dict, dict):
        free_cfg_dict = free_cfg_dict.copy()

        # --- aliases for backward / alternate naming ---
        alias_map = {
            "lcfs_contour_selector": "lcfs_selector",
            "tol_Ip_change": "tol_Ip_rel_change",
            "tol_lcfs": "tol_lcfs_change",
            "alpha_lcfs": "under_relax_lcfs",
            "alpha_plasma": "under_relax_plasma",
            "max_iter": "max_outer_iter",
        }

        for old, new in list(alias_map.items()):
            if old in free_cfg_dict and new not in free_cfg_dict:
                free_cfg_dict[new] = free_cfg_dict.pop(old)
                logger.info(
                    "Free-boundary config: mapped alias '%s' → '%s'", old, new
                )

        # --- drop & log unknown keys ---
        allowed = set(FreeBoundaryConfig.__dataclass_fields__.keys())
        unknown_keys = [k for k in free_cfg_dict.keys() if k not in allowed]

        for k in unknown_keys:
            logger.warning(
                "Free-boundary config: ignoring unsupported key '%s' (value=%r)",
                k, free_cfg_dict[k]
            )

        free_cfg_dict = {k: v for k, v in free_cfg_dict.items() if k in allowed}

    else:
        logger.warning(
            "free_boundary config is not a dict (%s); using defaults",
            type(free_cfg_dict).__name__,
        )
        free_cfg_dict = {}

    free_cfg = FreeBoundaryConfig(**free_cfg_dict)

    # Read HDF5 inputs and run solver
    with h5py.File(h5_path, "a") as h5:
        # Required datasets
        R = h5_read_array(h5, "/grid/R")
        Z = h5_read_array(h5, "/grid/Z")
        RR = h5_read_array(h5, "/grid/RR")
        ZZ = h5_read_array(h5, "/grid/ZZ")

        G_psi = h5_read_array(h5, "/device/coil_greens/psi_per_amp")
        I_pf = h5_read_array(h5, "/device/coils/I_pf")

        lcfs_init = h5_read_array(h5, "/target/boundary")

        # psi_lcfs (prefer stored)
        try:
            psi_lcfs = float(h5_read_scalar(h5, "/target/psi_boundary"))
        except Exception:
            psi_lcfs = float((target_cfg.get("global_targets") or {}).get("psi_lcfs", 0.0))

        # Optional initial guess from stage 03 (new path)
        psi_plasma_init = None
        if args.use_fixed_init:
            try:
                psi_plasma_init = h5_read_array(h5, "/equilibrium/fixed/psi")
                logger.info("Using /equilibrium/fixed/psi as psi_plasma_init")
            except Exception:
                logger.info("No /equilibrium/fixed/psi found; psi_plasma_init=None (zeros).")

        logger.info("Calling solve_free_boundary_equilibrium ...")
        out = solve_free_boundary_equilibrium(
            R=R, Z=Z, RR=RR, ZZ=ZZ,
            G_psi=G_psi,
            I_pf=I_pf,
            psi_lcfs=psi_lcfs,
            lcfs_init=lcfs_init,
            profile=profile,
            fixed_cfg=fixed_cfg,
            free_cfg=free_cfg,
            psi_plasma_init=psi_plasma_init,
        )

        eq = out["equilibrium"]
        hist = out["history"]

        # -------------------------
        # Write equilibrium outputs
        # -------------------------
        logger.info("Writing /equilibrium/free outputs ...")
        h5_ensure_group(h5, "/equilibrium")
        h5_ensure_group(h5, "/equilibrium/free")
        h5_ensure_group(h5, "/equilibrium/free/axis")
        h5_ensure_group(h5, "/equilibrium/free/history")
        h5_ensure_group(h5, "/equilibrium/free/convergence")

        h5_write_array(h5, "/equilibrium/free/psi_total", np.asarray(eq["psi_total"]))
        h5_write_array(h5, "/equilibrium/free/psi_plasma", np.asarray(eq["psi_plasma"]))
        h5_write_array(h5, "/equilibrium/free/psi_vac", np.asarray(eq["psi_vac"]))
        h5_write_array(h5, "/equilibrium/free/lcfs_poly", np.asarray(eq["lcfs_poly"]))

        axis = eq.get("axis_RZ", None)
        if axis is not None:
            h5_write_scalar(h5, "/equilibrium/free/axis/R", float(axis[0]))
            h5_write_scalar(h5, "/equilibrium/free/axis/Z", float(axis[1]))
        if "psi_axis" in eq and np.isfinite(eq["psi_axis"]):
            h5_write_scalar(h5, "/equilibrium/free/psi_axis", float(eq["psi_axis"]))
        h5_write_scalar(h5, "/equilibrium/free/psi_lcfs", float(psi_lcfs))

        # If the solver provides these (recommended), store them under /equilibrium/free
        if eq.get("p_psi", None) is not None:
            h5_write_array(h5, "/equilibrium/free/p_psi", np.asarray(eq["p_psi"]))
        if eq.get("F_psi", None) is not None:
            h5_write_array(h5, "/equilibrium/free/F_psi", np.asarray(eq["F_psi"]))
        if eq.get("mask", None) is not None:
            h5_write_array(h5, "/equilibrium/free/plasma_mask", np.asarray(eq["mask"]).astype(np.uint8),
                           attrs={"note": "1=inside plasma"})
        if eq.get("jphi", None) is not None:
            h5_write_array(h5, "/equilibrium/free/jphi", np.asarray(eq["jphi"]))

        # History / convergence (stage-05 bookkeeping)
        h5_write_array(h5, "/equilibrium/free/history/outer_iter", np.asarray(hist["outer_iter"]))
        h5_write_array(h5, "/equilibrium/free/history/lcfs_rms", np.asarray(hist["lcfs_rms"]))
        h5_write_array(h5, "/equilibrium/free/history/Ip", np.asarray(hist["Ip"]))
        h5_write_array(h5, "/equilibrium/free/history/dIp_rel", np.asarray(hist["dIp_rel"]))
        h5_write_array(h5, "/equilibrium/free/history/alpha_lcfs", np.asarray(hist["alpha_lcfs"]))
        h5_write_array(h5, "/equilibrium/free/history/alpha_plasma", np.asarray(hist["alpha_plasma"]))

        h5_write_scalar(h5, "/equilibrium/free/convergence/converged", bool(hist["converged"]))
        h5_write_scalar(h5, "/equilibrium/free/convergence/n_outer_iter", int(hist["n_outer_iter"]))
        h5_write_scalar(h5, "/equilibrium/free/convergence/stop_reason", str(hist["stop_reason"]))

        # --------------------------------------------
        # Optional: fields + derived
        # --------------------------------------------
        if args.no_fields:
            logger.info("--no-fields set: skipping fields/derived.")
        else:
            psi_total = np.asarray(eq["psi_total"])

            h5_ensure_group(h5, "/equilibrium/free/fields")
            h5_ensure_group(h5, "/equilibrium/free/derived")

            # BR, BZ from psi
            try:
                from tokdesign.physics._fields import compute_BR_BZ_from_psi
                BR, BZ = compute_BR_BZ_from_psi(R, Z, psi_total)
                h5_write_array(h5, "/equilibrium/free/fields/BR", np.asarray(BR), attrs={"units": "T"})
                h5_write_array(h5, "/equilibrium/free/fields/BZ", np.asarray(BZ), attrs={"units": "T"})
            except Exception as e:
                logger.warning("Failed computing BR/BZ: %s", e)

            # Bphi optional (depends on your F(psi) implementation)
            try:
                from tokdesign.physics._gs_profiles import normalize_psi, F_of_psin
                from tokdesign.physics._fields import compute_Bphi_from_F

                psi_axis = float(eq.get("psi_axis", np.nan))
                if not np.isfinite(psi_axis):
                    raise RuntimeError("psi_axis missing; cannot compute Bphi.")

                psin = normalize_psi(psi_total, psi_axis=psi_axis, psi_lcfs=psi_lcfs, clip=True)
                F = F_of_psin(psin, profile)
                Bphi = compute_Bphi_from_F(RR, F)
                h5_write_array(h5, "/equilibrium/free/fields/Bphi", np.asarray(Bphi), attrs={"units": "T"})
            except Exception as e:
                logger.info("Skipping Bphi (optional): %s", e)

            # Derived Ip (requires jphi + mask returned by solver)
            try:
                from tokdesign.physics._derived import compute_Ip
                jphi = eq.get("jphi", None)
                mask = eq.get("mask", None)
                if jphi is not None and mask is not None:
                    dR = float(R[1] - R[0])
                    dZ = float(Z[1] - Z[0])
                    Ip = float(compute_Ip(np.asarray(jphi), dR, dZ, np.asarray(mask)))
                    h5_write_scalar(h5, "/equilibrium/free/derived/Ip", Ip, attrs={"units": "A"})
            except Exception as e:
                logger.warning("Failed computing /equilibrium/free/derived/Ip: %s", e)

        logger.info("Stage 05 complete. converged=%s n_outer_iter=%s",
                    hist["converged"], hist["n_outer_iter"])


if __name__ == "__main__":
    main()
