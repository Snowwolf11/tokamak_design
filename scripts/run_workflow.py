#!/usr/bin/env python3
"""
run_workflow.py
===============

Convenience driver to run multiple pipeline steps in sequence without manually
passing the run directory around.

It orchestrates:
  00_init_run.py
  01_build_device.py
  (later: 02_target_boundary.py, 03_solve_fixed_gs.py, ...)

Why
---
• One command to run the whole workflow up to a chosen stage
• Keeps run directory handling consistent
• Captures all CLI args and logs them in the run folder

Usage
-----
python scripts/run_workflow.py \
  --device configs/baseline_device.yaml \
  --target configs/target_equilibrium.yaml \
  --solver configs/solver.yaml \
  --run-root data/runs \
  --prefix baseline \
  --notes "test" \
  --to 01

Stages
------
--to accepts:
  00, 01, 02, 03, 04, 05, 06, 07
Currently implemented here: 00 and 01 (others can be added later).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# ------------------------------------------------------------
# Map stage -> script filename
# ------------------------------------------------------------
STEPS = {
    "00": "00_init_run.py",
    "01": "01_build_device.py",
    # Future:
    # "02": "02_target_boundary.py",
    # "03": "03_solve_fixed_gs.py",
    # "04": "04_fit_pf_currents.py",
    # "05": "05_solve_free_gs.py",
    # "06": "06_analyze.py",
    # "07": "07_make_report.py",
}

STEP_ORDER = ["00", "01"]  # extend as you implement


def _run_subprocess(cmd: List[str]) -> None:
    """
    Run a subprocess and stream stdout/stderr live.
    Raises if the command fails.
    """
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _parse_run_dir_from_init_output(stdout_text: str) -> Optional[Path]:
    """
    Optional helper if you later capture output.
    Currently we don't capture (we stream), so this is unused.
    """
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokamak_design workflow driver.")
    parser.add_argument("--device", required=True, help="baseline_device.yaml")
    parser.add_argument("--target", required=True, help="target_equilibrium.yaml")
    parser.add_argument("--solver", required=True, help="solver.yaml")
    parser.add_argument("--run-root", default="data/runs", help="run root dir")
    parser.add_argument("--prefix", default="", help="run id suffix/prefix")
    parser.add_argument("--notes", default="", help="notes to store in /meta/notes")
    parser.add_argument("--log-level", default="INFO", help="log level")
    parser.add_argument("--to", default="01", help="final stage to run (e.g. 00, 01)")
    parser.add_argument("--create-placeholders", action="store_true",
                        help="Pass through to 00_init_run.py")
    parser.add_argument("--overwrite", action="store_true",
                        help="Pass through to later scripts that support overwrite")
    args = parser.parse_args()

    # Validate stage target
    to_stage = args.to.zfill(2)
    if to_stage not in STEPS:
        raise ValueError(f"Unknown stage '{args.to}'. Known: {sorted(STEPS.keys())}")

    # Resolve script directory
    scripts_dir = Path(__file__).resolve().parent

    # -----------------------------
    # Step 00: init run
    # -----------------------------
    # We want to capture the run_id / run_dir produced by 00_init_run.py.
    # Simplest robust approach: reproduce the run_id generation here? Not ideal.
    # Better: 00_init_run.py writes the run_dir deterministically and prints it,
    # and we can "discover" the most recently created run under run-root+prefix.
    #
    # Most robust approach: modify 00_init_run.py to accept --run-id (optional)
    # and return it, but for now we'll do a safe discovery strategy.
    #
    # Strategy: run 00_init_run.py, then find newest directory in run-root
    # matching "*_{prefix}" if prefix provided, else newest overall.
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    init_script = scripts_dir / STEPS["00"]
    init_cmd = [
        sys.executable, str(init_script),
        "--device", args.device,
        "--target", args.target,
        "--solver", args.solver,
        "--run-root", str(run_root),
        "--prefix", args.prefix,
        "--notes", args.notes,
        "--log-level", args.log_level,
    ]
    if args.create_placeholders:
        init_cmd.append("--create-placeholders")

    _run_subprocess(init_cmd)

    # Discover run_dir created by step 00 (newest directory under run_root)
    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {run_root} after init step.")
    run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)

    print(f"\n>>> Detected run_dir: {run_dir}")

    # If only running to 00, stop here
    if to_stage == "00":
        print("\nWorkflow completed up to stage 00.")
        print(f"  run_dir: {run_dir}")
        return

    # -----------------------------
    # Step 01: build device
    # -----------------------------
    build_script = scripts_dir / STEPS["01"]
    # Prefer archived config inside run_dir if it exists
    archived_device = run_dir / "inputs" / Path(args.device).name
    device_for_step = archived_device if archived_device.exists() else Path(args.device).expanduser().resolve()

    step01_cmd = [
        sys.executable, str(build_script),
        "--run-dir", str(run_dir),
        "--device", str(device_for_step),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        step01_cmd.append("--overwrite")

    _run_subprocess(step01_cmd)

    print("\nWorkflow completed up to stage 01.")
    print(f"  run_dir: {run_dir}")


if __name__ == "__main__":
    main()
