#!/usr/bin/env python3
"""
02_target_boundary.py
=====================

Create the target plasma boundary (LCFS) and store it in results.h5.

Responsibilities
----------------
• Read target_equilibrium.yaml
• Generate a target boundary polyline (LCFS) using plasma_geometry settings
  (v1: Miller parameterization)
• Store target boundary and target scalar values in:
    /target/boundary
    /target/psi_lcfs
    /target/global_targets/*

Why this exists
---------------
• Keeps "desired plasma shape" separate from device geometry
• Provides a consistent target for:
    - fixed-boundary GS (03)
    - coil current fitting (04)
    - free-boundary GS initial guess (05)

Outputs (results.h5)
--------------------
/target/
  boundary              (N,2) polyline in (R,Z) [m]
  psi_lcfs              scalar [Wb/rad] (or your chosen convention)
  global_targets/
    enforce             list of strings (what targets are enforced)
    Ip_target           scalar [A] (optional target; may be NaN if unused)

Schema notes
------------
Your schema mentions both psi_boundary and psi_lcfs in different drafts.
This script writes /target/psi_lcfs (as requested here) AND also writes
/target/psi_boundary as an alias for robustness, unless you disable it.

Usage
-----
python scripts/02_target_boundary.py \
  --run-dir data/runs/<RUN_ID> \
  --target configs/target_equilibrium.yaml
"""

from pathlib import Path
import argparse
from typing import Dict, Any, List

import numpy as np
import yaml
import h5py

from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.h5 import (
    h5_ensure_group,
    h5_write_array,
    h5_write_scalar,
    h5_write_strings,
    h5_write_dict_as_attrs,
)
from tokdesign.io.schema import validate_h5_structure
from tokdesign.geometry.plasma_boundary import miller_boundary, boundary_kappa_delta, boundary_area


# ============================================================
# HELPERS
# ============================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _assert_run_dir(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    results = run_dir / "results.h5"
    if not results.exists():
        raise FileNotFoundError(
            f"results.h5 not found in run directory.\nExpected: {results}\nDid you run 00_init_run.py?"
        )
    return results


def _read_schema_version(results_path: Path) -> str:
    with h5py.File(results_path, "r") as h5:
        sv = h5["/meta/schema_version"][()]
    if isinstance(sv, (bytes, np.bytes_)):
        return sv.decode("utf-8")
    return str(sv)


def _normalize_enforce_list(enforce) -> List[str]:
    """
    Accept enforce in YAML as:
      - list of strings, e.g. ["Ip_target"]
      - dict of {name: bool}, e.g. {Ip_target: true, beta_p: false}
      - single string
    Return list[str].
    """
    if enforce is None:
        return []
    if isinstance(enforce, str):
        return [enforce]
    if isinstance(enforce, list):
        return [str(x) for x in enforce]
    if isinstance(enforce, dict):
        return [str(k) for k, v in enforce.items() if bool(v)]
    raise ValueError("global_targets.enforce must be a list, dict, string, or null.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Create target LCFS boundary and store in results.h5.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory created by 00_init_run.py")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target_equilibrium.yaml (prefer archived copy in run_dir/inputs)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing /target datasets")
    parser.add_argument("--write-psi-lcfs-alias", action="store_true",
                        help="Also write /target/psi_lcfs as alias of /target/psi_boundary")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    results_path = _assert_run_dir(run_dir)

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path, level=args.log_level)
    logger.info("Running 02_target_boundary.py")
    logger.info("Run dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))

    target_yaml = Path(args.target).expanduser().resolve()
    if not target_yaml.exists():
        raise FileNotFoundError(f"Target YAML not found: {target_yaml}")

    cfg = _load_yaml(target_yaml)

    # -----------------------------
    # Parse plasma geometry
    # -----------------------------
    pg = cfg.get("plasma_geometry", {}) or {}
    gtype = str(pg.get("type", "miller")).lower()
    if gtype not in ("miller",):
        raise ValueError(f"Unsupported plasma_geometry.type='{gtype}'. v1 supports 'miller' only.")

    R0 = float(pg["R0"])
    a = float(pg["a"])
    kappa = float(pg.get("kappa", 1.0))
    delta = float(pg.get("delta", 0.0))
    Z0 = float(pg.get("Z0", 0.0))
    npts = int(pg.get("n_boundary_points", 400))

    # Create Miller boundary centered at Z=0, then shift by Z0
    boundary = miller_boundary(R0=R0, a=a, kappa=kappa, delta=delta, npts=npts)
    boundary[:, 1] += Z0  # shift Z

    # Quick derived metrics (useful for debugging and later analysis)
    area = boundary_area(boundary)
    kappa_est, delta_est = boundary_kappa_delta(boundary)

    logger.info("Target boundary: npts=%d (closed polyline length=%d)", npts, boundary.shape[0])
    logger.info("Target geometry (input): R0=%g a=%g kappa=%g delta=%g Z0=%g", R0, a, kappa, delta, Z0)
    logger.info("Target geometry (estimated): kappa=%g delta=%g area=%g m^2", kappa_est, delta_est, area)

    # -----------------------------
    # Parse global targets
    # -----------------------------
    gt = cfg.get("global_targets", {}) or {}

    # psi_lcfs is the target value of psi on the LCFS. You can choose it as 0.0
    # by convention for vacuum coil fitting. Keep it configurable.
    psi_lcfs = float(gt.get("psi_lcfs", 0.0))

    enforce_list = _normalize_enforce_list(gt.get("enforce", []))

    # Optional global targets
    Ip_target = gt.get("Ip_target", np.nan)
    try:
        Ip_target = float(Ip_target)
    except Exception:
        Ip_target = np.nan

    # -----------------------------
    # Write to HDF5
    # -----------------------------
    schema_version = _read_schema_version(results_path)

    with h5py.File(results_path, "r+") as h5:
        if not args.overwrite:
            if "/target/boundary" in h5 or "/target/psi_lcfs" in h5:
                raise RuntimeError(
                    "It looks like /target already exists in results.h5.\n"
                    "Refusing to overwrite. Re-run with --overwrite if intended."
                )

        h5_ensure_group(h5, "/target")
        h5_ensure_group(h5, "/target/global_targets")

        # Boundary polyline
        h5_write_array(h5, "/target/boundary", boundary, attrs={"units": "m"})

        # Scalar psi targets
        h5_write_scalar(h5, "/target/psi_boundary", psi_lcfs, attrs={"units": "Wb_per_rad"})
        if args.write_psi_lcfs_alias:
            h5_write_scalar(h5, "/target/psi_lcfs", psi_lcfs, attrs={"units": "Wb_per_rad"})

        # Global target bookkeeping
        h5_write_strings(h5, "/target/global_targets/enforce", enforce_list)
        h5_write_scalar(h5, "/target/global_targets/Ip_target", Ip_target, attrs={"units": "A"})

        # Store some helpful geometry attrs for quick inspection
        h5_write_dict_as_attrs(h5, "/target", {
            "plasma_geometry_type": gtype,
            "R0": R0, "a": a, "kappa_in": kappa, "delta_in": delta, "Z0": Z0,
            "kappa_est": kappa_est, "delta_est": delta_est,
            "area_m2": area,
        }, overwrite=True)

    logger.info("Wrote /target/* to results.h5")

    # Validate schema stage "target"
    validate_h5_structure(results_path, schema_version=schema_version, stage="target")
    logger.info("Schema validation passed for stage='target'")

    print("\nTarget boundary complete.")
    print(f"  run_dir:     {run_dir}")
    print(f"  results.h5:  {results_path}")
    print(f"  boundary:    {boundary.shape[0]} points (closed)")
    print(f"  psi_lcfs:    {psi_lcfs}")
    print(f"  enforce:     {enforce_list}")
    print("")


if __name__ == "__main__":
    main()
