#!/usr/bin/env python3
"""
01_build_device.py
==================

Build the "device" description for a run:
• create computational R-Z grid
• build vessel boundary geometry
• build PF coil list (including optional central solenoid)
• optionally precompute coil psi Green's functions on the grid

This script assumes you already ran:
    00_init_run.py
so the run directory exists and results.h5 contains /meta/*.

Outputs (results.h5)
--------------------
/grid/
  R, Z, RR, ZZ

/device/
  vessel_boundary
  coils/
    names
    centers
    radii
    I_max
    I_pf         (initial coil currents; later updated by fitting/optimization)
  coil_greens/
    psi_per_amp  (Nc, NZ, NR)  if enabled

Also writes some useful metadata attributes where helpful.

Validation
----------
At the end, validates HDF5 structure for schema stage "device".

Usage
-----
python scripts/01_build_device.py \
  --run-dir data/runs/<RUN_ID> \
  --device configs/baseline_device.yaml

Notes
-----
• The device YAML path can be inside or outside the run folder; the script will
  read the provided path. (The archived copy is in run_dir/inputs for provenance.)
• For reproducibility, you typically pass the archived config path:
    --device data/runs/<RUN_ID>/inputs/baseline_device.yaml
"""

from pathlib import Path
import argparse
from typing import Dict, Any

import numpy as np
import yaml
import h5py

from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.h5 import (
    open_h5,
    h5_ensure_group,
    h5_write_array,
    h5_write_scalar,
    h5_write_strings,
    h5_write_dict_as_attrs,
)
from tokdesign.io.schema import validate_h5_structure

from tokdesign.geometry.grid import make_rz_grid
from tokdesign.geometry.vessel import load_vessel_from_config
from tokdesign.geometry.coils import (
    coils_from_config,
    coil_centers,
    coil_currents,
    coil_current_limits,
    compute_coil_psi_greens,
)


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
            f"results.h5 not found in run directory.\n"
            f"Expected: {results}\n"
            f"Did you run 00_init_run.py?"
        )
    return results


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build device geometry and coil greens for a run.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory created by 00_init_run.py")
    parser.add_argument("--device", type=str, required=True,
                        help="Path to baseline_device.yaml (prefer the archived copy in run_dir/inputs)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing /grid and /device datasets in results.h5")

    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    results_path = _assert_run_dir(run_dir)

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path, level=args.log_level)
    logger.info("Running 01_build_device.py")
    logger.info("Run dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))

    device_yaml = Path(args.device).expanduser().resolve()
    if not device_yaml.exists():
        raise FileNotFoundError(f"Device YAML not found: {device_yaml}")

    cfg = _load_yaml(device_yaml)

    # -----------------------------
    # Read grid section
    # -----------------------------
    grid = cfg.get("grid", {}) or {}
    R_min = float(grid["R_min"])
    R_max = float(grid["R_max"])
    Z_min = float(grid["Z_min"])
    Z_max = float(grid["Z_max"])
    NR = int(grid["NR"])
    NZ = int(grid["NZ"])

    logger.info("Grid: R=[%g, %g], Z=[%g, %g], NR=%d, NZ=%d", R_min, R_max, Z_min, Z_max, NR, NZ)

    R, Z, RR, ZZ = make_rz_grid(R_min, R_max, Z_min, Z_max, NR, NZ)

    # -----------------------------
    # Vessel geometry
    # -----------------------------
    vessel_boundary = load_vessel_from_config(cfg)
    logger.info("Vessel boundary points: %d", vessel_boundary.shape[0])

    # -----------------------------
    # Coils
    # -----------------------------
    coils = coils_from_config(cfg, include_solenoid=True)
    Nc = len(coils)
    centers = coil_centers(coils)            # (Nc, 2)
    radii = np.array([c.a for c in coils])   # (Nc,)
    I_init = coil_currents(coils)            # (Nc,)
    I_max = coil_current_limits(coils)       # (Nc,)
    names = [c.name for c in coils]

    logger.info("Loaded %d coils: %s", Nc, ", ".join(names))

    # -----------------------------
    # Numerics settings
    # -----------------------------
    numerics = cfg.get("numerics", {}) or {}
    precompute = bool(numerics.get("precompute_coil_greens", True))
    greens_method = str(numerics.get("greens_method", "analytic_elliptic"))

    # -----------------------------
    # Write to HDF5
    # -----------------------------
    with h5py.File(results_path, "r+" if results_path.exists() else "w") as h5:

        # Optional overwrite behavior:
        # Our h5_write_* already overwrites datasets by default.
        # But sometimes you want to guard against accidentally re-running.
        if not args.overwrite:
            # If /grid already has R dataset, assume device step already done.
            if "/grid/R" in h5 or "/device/vessel_boundary" in h5:
                raise RuntimeError(
                    "It looks like /grid or /device already exists in results.h5.\n"
                    "Refusing to overwrite. Re-run with --overwrite if you intend to replace them."
                )

        # Ensure base groups exist
        h5_ensure_group(h5, "/grid")
        h5_ensure_group(h5, "/device")
        h5_ensure_group(h5, "/device/coils")

        # Write grid
        h5_write_array(h5, "/grid/R", R, attrs={"units": "m"})
        h5_write_array(h5, "/grid/Z", Z, attrs={"units": "m"})
        h5_write_array(h5, "/grid/RR", RR, attrs={"units": "m"})
        h5_write_array(h5, "/grid/ZZ", ZZ, attrs={"units": "m"})

        # Store grid config as group attributes (handy for quick inspection)
        h5_write_dict_as_attrs(h5, "/grid", {
            "R_min": R_min, "R_max": R_max, "Z_min": Z_min, "Z_max": Z_max, "NR": NR, "NZ": NZ
        }, overwrite=True)

        # Vessel boundary
        h5_write_array(h5, "/device/vessel_boundary", vessel_boundary, attrs={"units": "m"})

        # Coils datasets
        h5_write_strings(h5, "/device/coils/names", names)
        h5_write_array(h5, "/device/coils/centers", centers, attrs={"units": "m"})
        h5_write_array(h5, "/device/coils/radii", radii, attrs={"units": "m"})
        h5_write_array(h5, "/device/coils/I_max", I_max, attrs={"units": "A"})
        # IMPORTANT: store the *current vector* (initial guess) as I_pf
        h5_write_array(h5, "/device/coils/I_pf", I_init, attrs={"units": "A"})

        # Record coil list metadata
        h5_write_dict_as_attrs(h5, "/device/coils", {
            "Nc": Nc,
            "include_solenoid": True,
        }, overwrite=True)

        # Precompute coil greens (psi per amp)
        if precompute:
            logger.info("Computing coil psi Green's functions (method=%s)...", greens_method)
            G_psi = compute_coil_psi_greens(coils, RR, ZZ, method=greens_method)
            logger.info("Computed G_psi shape: %s", str(G_psi.shape))

            h5_ensure_group(h5, "/device/coil_greens")
            h5_write_array(
                h5,
                "/device/coil_greens/psi_per_amp",
                G_psi,
                attrs={"units": "Wb_per_rad_per_A", "method": greens_method},
            )

        else:
            logger.info("Skipping coil greens precompute (numerics.precompute_coil_greens=false)")

        # Store toroidal field parameters (not used yet, but belongs to device)
        tf = cfg.get("toroidal_field", {}) or {}
        if tf:
            h5_ensure_group(h5, "/device/toroidal_field")
            h5_write_scalar(h5, "/device/toroidal_field/R0", float(tf.get("R0", np.nan)), attrs={"units": "m"})
            h5_write_scalar(h5, "/device/toroidal_field/B0", float(tf.get("B0", np.nan)), attrs={"units": "T"})

    logger.info("Wrote /grid and /device data to results.h5")

    # Validate schema stage "device"
    # Schema version is stored in /meta; we validate against that.
    with h5py.File(results_path, "r") as h5:
        schema_version = h5["/meta/schema_version"][()]
        if isinstance(schema_version, (bytes, np.bytes_)):
            schema_version = schema_version.decode("utf-8")
        else:
            schema_version = str(schema_version)

    validate_h5_structure(results_path, schema_version=schema_version, stage="device")
    logger.info("Schema validation passed for stage='device'")

    print("\nDevice build complete.")
    print(f"  run_dir:     {run_dir}")
    print(f"  results.h5:  {results_path}")
    print(f"  coils:       {Nc}")
    print(f"  greens:      {'yes' if precompute else 'no'}")
    print("")


if __name__ == "__main__":
    main()
