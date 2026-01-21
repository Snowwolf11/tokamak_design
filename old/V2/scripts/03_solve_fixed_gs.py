#!/usr/bin/env python3
"""
03_solve_fixed_gs.py
====================

Fixed-boundary Grad–Shafranov equilibrium solve (pure plasma physics).

Reads
-----
• results.h5 (must already contain /grid and /target from upstream steps)
• target_equilibrium.yaml
• solver.yaml

Uses from target_equilibrium.yaml
--------------------------------
• profiles:
    - pressure.model, p0, alpha_p
    - toroidal_field_function.model, F0, alpha_F
• global_targets:
    - Ip_target
    - enforce
    - psi_lcfs (boundary flux convention)
• diagnostics:
    - compute_q_profile ("none" or "approx")
    - compute_beta_p (bool)
    - compute_li (bool)

Uses from solver.yaml
--------------------
fixed_boundary_gs:
  - max_picard_iter
  - tol_rel_change
  - under_relaxation
  - linear_solver
  - cg_tol
  - cg_max_iter
  - clip_psin

Requires in results.h5
----------------------
/grid/R, /grid/Z, /grid/RR, /grid/ZZ
/target/boundary
/target/psi_boundary

Writes to results.h5
--------------------
/equilibrium/psi
/equilibrium/psi_axis
/equilibrium/psi_lcfs
/equilibrium/p_psi
/equilibrium/F_psi
/equilibrium/plasma_mask
/equilibrium/jphi

/fields/BR
/fields/BZ
/fields/Bphi

/derived/Ip
/derived/beta_p
/derived/li
/derived/kappa
/derived/delta
/derived/q_profile

Validation
----------
- Validates prerequisites: stage="device" and stage="target"
- Validates outputs: stage="fixed_eq"

Notes
-----
- v1: we do NOT yet enforce Ip_target during the solve; we record Ip and keep hooks for
  later scaling/constraint enforcement.
- This step does NOT depend on coil geometry.
"""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Any, Dict

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
)
from tokdesign.io.schema import validate_h5_structure

from tokdesign.geometry.grid import grid_spacing
from tokdesign.geometry._plasma_boundary import boundary_kappa_delta, ensure_closed_polyline

from tokdesign.physics._gs_profiles import profiles_from_dict, normalize_psi
from tokdesign.physics._gs_solve_fixed import solve_fixed_boundary_gs
from tokdesign.physics._fields import compute_BR_BZ_from_psi, compute_Bphi_from_F
from tokdesign.physics._derived import (
    compute_Ip,
    compute_beta_p,
    compute_li,
    approx_q_profile,
)


# ============================================================
# HELPERS
# ============================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _require_file(path: Path, what: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-boundary Grad–Shafranov equilibrium solve.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory created by 00_init_run.py (contains results.h5)")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target_equilibrium.yaml (prefer archived copy in run_dir/inputs)")
    parser.add_argument("--solver", type=str, required=True,
                        help="Path to solver.yaml (prefer archived copy in run_dir/inputs)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing /equilibrium, /fields, /derived datasets.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    assert_is_run_dir(run_dir)

    results_path = run_dir / "results.h5"
    if not results_path.exists():
        raise FileNotFoundError(f"results.h5 not found in run directory: {results_path}")

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path, level=args.log_level)

    target_yaml = _require_file(Path(args.target), "Target YAML")
    solver_yaml = _require_file(Path(args.solver), "Solver YAML")

    logger.info("Running 03_solve_fixed_gs.py")
    logger.info("Run dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))
    logger.info("target.yaml: %s", str(target_yaml))
    logger.info("solver.yaml: %s", str(solver_yaml))

    # Validate prerequisites (device + target must exist)
    with open_h5(results_path, "r") as h5:
        schema_version = h5_read_scalar(h5, "/meta/schema_version")
    validate_h5_structure(results_path, schema_version=schema_version, stage="device")
    validate_h5_structure(results_path, schema_version=schema_version, stage="target")

    target_cfg = _load_yaml(target_yaml)
    solver_cfg_all = _load_yaml(solver_yaml)
    solver_cfg = (solver_cfg_all.get("fixed_boundary_gs", {}) or {}).copy()

    # Diagnostics settings
    diag = target_cfg.get("diagnostics", {}) or {}
    compute_q = str(diag.get("compute_q_profile", "none")).lower()
    compute_beta = bool(diag.get("compute_beta_p", True))
    compute_li_flag = bool(diag.get("compute_li", True))

    # Global targets (recorded only in v1)
    gt = target_cfg.get("global_targets", {}) or {}
    enforce = str(gt.get("enforce", "none")).lower()
    Ip_target = float(gt.get("Ip_target", np.nan))
    psi_lcfs_cfg = gt.get("psi_lcfs", None)

    # Profiles (YAML-aligned dispatcher)
    profiles_cfg = (target_cfg.get("profiles", {}) or {}).copy()
    profiles = profiles_from_dict(profiles_cfg)

    # Read grid + target boundary from results.h5
    with open_h5(results_path, "r") as h5:
        R = h5_read_array(h5, "/grid/R").astype(float)
        Z = h5_read_array(h5, "/grid/Z").astype(float)
        RR = h5_read_array(h5, "/grid/RR").astype(float)
        ZZ = h5_read_array(h5, "/grid/ZZ").astype(float)

        lcfs_poly = h5_read_array(h5, "/target/boundary").astype(float)
        lcfs_poly = ensure_closed_polyline(lcfs_poly)

        psi_lcfs_h5 = float(h5_read_scalar(h5, "/target/psi_boundary"))

    # psi_lcfs convention: prefer explicit target YAML if set, else use H5
    if psi_lcfs_cfg is not None:
        psi_lcfs = float(psi_lcfs_cfg)
    else:
        psi_lcfs = float(psi_lcfs_h5)

    logger.info("psi_lcfs: %g", psi_lcfs)
    logger.info("enforce: %s, Ip_target: %s", enforce, f"{Ip_target:g}" if np.isfinite(Ip_target) else "nan")

    # Refuse overwrite unless requested
    if not args.overwrite:
        with open_h5(results_path, "r") as h5:
            if (
                ("/equilibrium/fixed/psi" in h5)
                or ("/equilibrium/fixed/fields/BR" in h5)
                or ("/equilibrium/fixed/derived/Ip" in h5)
            ):
                raise RuntimeError(
                    "It looks like /equilibrium/fixed already exists in results.h5.\n"
                    "Refusing to overwrite. Re-run with --overwrite if intended."
                )

    # Solve fixed-boundary GS
    out = solve_fixed_boundary_gs(
        R=R, Z=Z, RR=RR, ZZ=ZZ,
        lcfs_poly=lcfs_poly,
        psi_lcfs=float(psi_lcfs),
        profiles=profiles,
        solver_cfg=solver_cfg,
        psi_init=None,
    )

    psi = out["psi"]
    psi_axis = float(out["psi_axis"])
    jphi = out["jphi"]
    plasma_mask = out["mask"]

    if out["converged"]:
        logger.info("Fixed-boundary GS converged in %d Picard iterations (final rel_change=%g).",
                    len(out["history"]["rel_change"]), out["history"]["rel_change"][-1])
    else:
        logger.warning("Fixed-boundary GS did NOT converge (final rel_change=%g).",
                       out["history"]["rel_change"][-1] if out["history"]["rel_change"] else np.nan)

    # Compute p(psi) and F(psi) arrays for storage/diagnostics
    psin = normalize_psi(psi, psi_axis, float(psi_lcfs), clip=True)
    p_psi = profiles.pressure.p(psin)
    F_psi = profiles.toroidal_field_function.F(psin)

    # Fields
    BR, BZ = compute_BR_BZ_from_psi(R, Z, psi)
    Bphi = compute_Bphi_from_F(RR, F_psi)

    # Derived
    dR, dZ = grid_spacing(R, Z)
    Ip = compute_Ip(jphi, dR, dZ, plasma_mask)
    kappa, delta = boundary_kappa_delta(lcfs_poly)

    beta_p = float("nan")
    li = float("nan")
    if compute_beta:
        beta_p = compute_beta_p(p_psi, plasma_mask, dR, dZ, Ip)
    if compute_li_flag:
        li = compute_li(BR, BZ, RR, plasma_mask, dR, dZ, Ip)

    q_profile = np.zeros((0, 2), dtype=float)
    if compute_q != "none":
        qp = approx_q_profile(
            R=R, Z=Z, psi=psi,
            BR=BR, BZ=BZ,
            profiles=profiles,
            psi_axis=psi_axis,
            psi_lcfs=float(psi_lcfs),
            axis_RZ=out["axis_RZ"],
        )
        q_profile = np.column_stack([qp["psin"], qp["q"]]).astype(float)

    # Write outputs
    with open_h5(results_path, "r+") as h5:
        h5_ensure_group(h5, "/equilibrium")
        h5_ensure_group(h5, "/equilibrium/fixed")
        h5_ensure_group(h5, "/equilibrium/fixed/fields")
        h5_ensure_group(h5, "/equilibrium/fixed/derived")

        # equilibrium (fixed)
        h5_write_array(h5, "/equilibrium/fixed/psi", psi, attrs={"units": "Wb_per_rad"})
        h5_write_scalar(h5, "/equilibrium/fixed/psi_axis", psi_axis, attrs={"units": "Wb_per_rad"})
        h5_write_scalar(h5, "/equilibrium/fixed/psi_lcfs", float(psi_lcfs), attrs={"units": "Wb_per_rad"})
        h5_write_array(h5, "/equilibrium/fixed/p_psi", p_psi, attrs={"units": "Pa"})
        h5_write_array(h5, "/equilibrium/fixed/F_psi", F_psi, attrs={"units": "T*m"})
        h5_write_array(h5, "/equilibrium/fixed/plasma_mask", plasma_mask.astype(np.uint8),
                       attrs={"note": "1=inside plasma"})
        h5_write_array(h5, "/equilibrium/fixed/jphi", jphi, attrs={"units": "A/m^2"})

        # fields (fixed)
        h5_write_array(h5, "/equilibrium/fixed/fields/BR", BR, attrs={"units": "T"})
        h5_write_array(h5, "/equilibrium/fixed/fields/BZ", BZ, attrs={"units": "T"})
        h5_write_array(h5, "/equilibrium/fixed/fields/Bphi", Bphi, attrs={"units": "T"})

        # derived (fixed)
        h5_write_scalar(h5, "/equilibrium/fixed/derived/Ip", float(Ip), attrs={"units": "A"})
        h5_write_scalar(h5, "/equilibrium/fixed/derived/beta_p", float(beta_p), attrs={"dimensionless": 1})
        h5_write_scalar(h5, "/equilibrium/fixed/derived/li", float(li), attrs={"dimensionless": 1})
        h5_write_scalar(h5, "/equilibrium/fixed/derived/kappa", float(kappa), attrs={"dimensionless": 1})
        h5_write_scalar(h5, "/equilibrium/fixed/derived/delta", float(delta), attrs={"dimensionless": 1})
        h5_write_array(h5, "/equilibrium/fixed/derived/q_profile", q_profile, attrs={"columns": "psin,q"})

        # provenance/debug attrs (fixed)
        h5_write_dict_as_attrs(h5, "/equilibrium/fixed", {
            "converged": bool(out["converged"]),
            "n_picard_iter": int(len(out["history"]["rel_change"])),
            "final_rel_change": float(out["history"]["rel_change"][-1]) if out["history"]["rel_change"] else np.nan,
            "enforce": enforce,
            "Ip_target": float(Ip_target),
        }, overwrite=True)

    # Validate outputs
    validate_h5_structure(results_path, schema_version=schema_version, stage="fixed_eq")

    logger.info("Wrote equilibrium/fields/derived to results.h5")
    logger.info("Derived: Ip=%g A, beta_p=%g, li=%g, kappa=%g, delta=%g", Ip, beta_p, li, kappa, delta)

    print("\nFixed-boundary GS solve complete.")
    print(f"  run_dir:    {run_dir}")
    print(f"  results.h5: {results_path}")
    print(f"  converged:  {out['converged']}")
    print(f"  Ip [A]:     {Ip:.6g}")
    print(f"  beta_p:     {beta_p:.6g}")
    print(f"  li:         {li:.6g}")
    print(f"  kappa:      {kappa:.6g}")
    print(f"  delta:      {delta:.6g}")
    if q_profile.size > 0:
        print(f"  q_profile:  N={q_profile.shape[0]} (stored in /derived/q_profile)")
    print("")


if __name__ == "__main__":
    main()
