#!/usr/bin/env python3
"""
00_init_run.py
==============

Initialize a new "run" directory and a fresh results.h5 file.

Responsibilities
----------------
• Create a new run directory structure:
    data/runs/<run_id>/
        inputs/
        figures/
        run.log
        results.h5
        summary.csv  (path reserved; not created here)

• Copy YAML configs into inputs/ for provenance
• Initialize results.h5 with /meta/* (and optional placeholder groups)
• Validate HDF5 structure for stage "init" using io/schema.py

Why this exists
---------------
This script makes every run self-contained and reproducible:
• configs are archived
• metadata is recorded in the HDF5 file
• later scripts can assume results.h5 exists and has /meta

Typical usage
-------------
python scripts/00_init_run.py \
  --device configs/baseline_device.yaml \
  --target configs/target_equilibrium.yaml \
  --solver configs/solver.yaml \
  --run-root data/runs \
  --prefix baseline \
  --notes "my first test"

Notes
-----
• This script does not create /grid, /device, /target contents (those come later).
  It may create placeholder groups to make browsing easier, but the schema validator
  for stage "init" only requires /meta/*.
"""

from pathlib import Path
from datetime import datetime
import argparse
import subprocess
from typing import Dict, Any

import yaml
import h5py

from tokdesign.io.paths import make_run_id, create_run_dir, copy_inputs
# IMPORTANT: do not name your module logging.py; use logging_utils.py
from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.h5 import h5_ensure_group, h5_write_scalar
from tokdesign.io.schema import validate_h5_structure


# ============================================================
# HELPERS
# ============================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML into a Python dict (safe loader)."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_git_commit(project_root: Path) -> str:
    """
    Best-effort git commit hash for provenance.
    Returns "UNKNOWN" if not in a git repo or git is unavailable.
    """
    try:
        # Use -C to run git in the project root
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def _infer_project_root() -> Path:
    """
    Infer project root as: scripts/../ (repo root).
    This assumes scripts/ is located at <repo>/scripts.
    """
    here = Path(__file__).resolve()
    return here.parent.parent


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a tokamak_design run directory and results.h5"
    )
    parser.add_argument("--device", type=str, required=True,
                        help="Path to baseline_device.yaml")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target_equilibrium.yaml")
    parser.add_argument("--solver", type=str, required=True,
                        help="Path to solver.yaml")
    parser.add_argument("--run-root", type=str, default="data/runs",
                        help="Root directory where runs are created (default: data/runs)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Optional run ID suffix (e.g. 'baseline', 'scan001')")
    parser.add_argument("--notes", type=str, default="",
                        help="Optional free-form notes stored in /meta/notes")
    parser.add_argument("--create-placeholders", action="store_true",
                        help="Create placeholder groups /grid, /device, /target in results.h5")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    args = parser.parse_args()

    # Resolve config paths
    device_path = Path(args.device).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()
    solver_path = Path(args.solver).expanduser().resolve()

    # Load YAMLs (for meta usage + early checks)
    device_cfg = _load_yaml(device_path)
    target_cfg = _load_yaml(target_path)
    solver_cfg = _load_yaml(solver_path)

    # Determine schema version:
    # Prefer device_cfg.meta.schema_version, else target, else solver, else "0.1"
    schema_version = (
        (device_cfg.get("meta", {}) or {}).get("schema_version")
        or (target_cfg.get("meta", {}) or {}).get("schema_version")
        or (solver_cfg.get("meta", {}) or {}).get("schema_version")
        or "0.1"
    )

    # Optional run label from meta.name (device preferred)
    # If you pass --prefix, that takes precedence.
    meta_name = (device_cfg.get("meta", {}) or {}).get("name", "")
    prefix = args.prefix.strip() or meta_name.strip() or ""

    # Create run directory structure
    run_id = make_run_id(prefix=prefix)
    run_root = Path(args.run_root).expanduser().resolve()
    paths = create_run_dir(run_root, run_id)

    # Setup logger in the new run dir
    logger = setup_logger(paths["log_path"], level=args.log_level)
    logger.info("Initialized run directory: %s", str(paths["run_dir"]))
    logger.info("Run ID: %s", run_id)

    # Copy configs into inputs/
    copy_inputs([device_path, target_path, solver_path], paths["inputs_dir"])
    logger.info("Copied configs to: %s", str(paths["inputs_dir"]))

    # Determine project root and git commit (best effort)
    project_root = _infer_project_root()
    git_commit = _get_git_commit(project_root)

    created_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Create results.h5 and write /meta/*
    results_path = paths["results_path"]
    with h5py.File(results_path, "w") as h5:
        h5_ensure_group(h5, "/meta")

        h5_write_scalar(h5, "/meta/schema_version", str(schema_version))
        h5_write_scalar(h5, "/meta/created_utc", created_utc)
        h5_write_scalar(h5, "/meta/run_id", run_id)
        h5_write_scalar(h5, "/meta/git_commit", git_commit)
        h5_write_scalar(h5, "/meta/notes", str(args.notes))

        # Optionally create placeholder groups (helps browsing early)
        if args.create_placeholders:
            for g in ["/grid", "/device", "/target", "/equilibrium", "/fields", "/derived", "/analysis", "/optimization"]:
                h5_ensure_group(h5, g)

    logger.info("Created HDF5 file: %s", str(results_path))

    # Validate init schema (ensures /meta/* exists)
    validate_h5_structure(results_path, schema_version=str(schema_version), stage="init")
    logger.info("Schema validation passed for stage='init'")

    # Print a simple “handoff” summary for the user
    print("\nRun created successfully.")
    print(f"  run_dir:      {paths['run_dir']}")
    print(f"  results.h5:   {paths['results_path']}")
    print(f"  inputs/:      {paths['inputs_dir']}")
    print(f"  figures/:     {paths['figures_dir']}")
    print(f"  run.log:      {paths['log_path']}")
    print(f"  schema:       {schema_version}")
    print(f"  git_commit:   {git_commit}")
    if args.notes:
        print(f"  notes:        {args.notes}")
    print("")


if __name__ == "__main__":
    main()
