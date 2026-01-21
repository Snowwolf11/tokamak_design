#!/usr/bin/env python3
"""
run_workflow.py
===============

High-level workflow driver for the tokamak_design pipeline.

Purpose
-------
This script orchestrates multiple *independent pipeline stages* in sequence,
without requiring the user to manually pass around the run directory or worry
about intermediate bookkeeping.

Each stage is implemented as a standalone script:
  • 00_init_run.py
  • 01_build_device.py
  • 02_target_boundary.py
  • 03_solve_fixed_gs.py
  • 04_fit_pf_currents.py
  • (later: 05_solve_free_gs.py, ...)

This driver:
  • runs the requested stages in the correct order
  • automatically detects the newly created run directory
  • always prefers archived configs in run_dir/inputs/
  • optionally runs quicklook at the end

Design Philosophy
-----------------
• Each stage script remains *fully usable on its own*
• This file contains *no physics or numerics*
• Stage-specific differences are isolated in small "command builder" functions
• The main loop is generic and does not need to change as new stages are added

Usage
-----
Run the workflow up to a given stage:

    python scripts/run_workflow.py \
        --device configs/baseline_device.yaml \
        --target configs/target_equilibrium.yaml \
        --solver configs/solver.yaml \
        --run-root data/runs \
        --prefix baseline \
        --notes "test run" \
        --use-fixed-init \
        --to 05 \
        --quicklook-fieldlines

Disable quicklook at the end:

    python scripts/run_workflow.py ... --no-quicklook

Typical Development Pattern
---------------------------
• Edit one stage (e.g. 03_solve_fixed_gs.py)
• Re-run:
      run_workflow.py --to 03 --overwrite
• Inspect results.h5 and figures/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional


# ============================================================
# STAGE REGISTRY
# ============================================================

# Order in which stages must be executed
# (this enforces dependencies)
STAGE_ORDER = ["00", "01", "02", "03", "05"]

# Mapping from stage code to script filename
STAGE_TO_SCRIPT = {
    "00": "00_init_run.py",
    "01": "01_build_device.py",
    "02": "02_target_boundary.py",
    "03": "03_solve_fixed_gs.py",
    "04": "04_fit_pf_currents.py",
    "05": "05_solve_free_gs.py",
}

# Quicklook script always runs at the end (unless disabled)
QUICKLOOK_SCRIPT = "09_quicklook.py"


# ============================================================
# GENERIC UTILITIES
# ============================================================

def _run_subprocess(cmd: List[str]) -> None:
    """
    Run a subprocess and stream stdout/stderr live.

    This uses subprocess.run(..., check=True) so any non-zero
    exit code immediately stops the workflow.
    """
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _detect_newest_run_dir(run_root: Path) -> Path:
    """
    Return the most recently modified run directory under run_root.

    This is how we discover the run directory created by 00_init_run.py.
    """
    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {run_root}.")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _archived_or_given(run_dir: Path, given_path: str) -> Path:
    """
    Prefer the archived config inside run_dir/inputs/ if it exists.

    This guarantees reproducibility when re-running later stages,
    even if the original config files have changed.
    """
    given = Path(given_path).expanduser().resolve()
    archived = run_dir / "inputs" / given.name
    return archived if archived.exists() else given


def _scripts_dir() -> Path:
    """Return the directory containing this script."""
    return Path(__file__).resolve().parent


# ============================================================
# STAGE-SPECIFIC COMMAND BUILDERS
# ============================================================

# Each function below constructs the exact subprocess command
# for a given stage. This isolates stage-specific logic.


def _cmd_00(args: argparse.Namespace, scripts_dir: Path, run_root: Path) -> List[str]:
    """
    Stage 00: initialize a new run directory and archive inputs.
    """
    script = scripts_dir / STAGE_TO_SCRIPT["00"]
    cmd = [
        sys.executable, str(script),
        "--device", args.device,
        "--target", args.target,
        "--solver", args.solver,
        "--run-root", str(run_root),
        "--prefix", args.prefix,
        "--notes", args.notes,
        "--log-level", args.log_level,
    ]
    if args.create_placeholders:
        cmd.append("--create-placeholders")
    return cmd


def _cmd_01(args: argparse.Namespace, scripts_dir: Path, run_dir: Path) -> List[str]:
    """
    Stage 01: build device geometry and grid.
    """
    script = scripts_dir / STAGE_TO_SCRIPT["01"]
    device_for_step = _archived_or_given(run_dir, args.device)
    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--device", str(device_for_step),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _cmd_02(args: argparse.Namespace, scripts_dir: Path, run_dir: Path) -> List[str]:
    """
    Stage 02: generate and store the target plasma boundary (LCFS).
    """
    script = scripts_dir / STAGE_TO_SCRIPT["02"]
    target_for_step = _archived_or_given(run_dir, args.target)
    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--target", str(target_for_step),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.write_psi_boundary_alias:
        cmd.append("--write-psi-boundary-alias")
    return cmd


def _cmd_03(args: argparse.Namespace, scripts_dir: Path, run_dir: Path) -> List[str]:
    """
    Stage 03: fixed-boundary Grad–Shafranov equilibrium solve.
    """
    script = scripts_dir / STAGE_TO_SCRIPT["03"]
    target_for_step = _archived_or_given(run_dir, args.target)
    solver_for_step = _archived_or_given(run_dir, args.solver)
    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--target", str(target_for_step),
        "--solver", str(solver_for_step),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd

def _cmd_04(args, scripts_dir, run_dir):
    script = scripts_dir / STAGE_TO_SCRIPT["04"]
    solver_for_step = _archived_or_given(run_dir, args.solver)

    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--solver", str(solver_for_step),
        "--log-level", args.log_level,
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd

def _cmd_05(args: argparse.Namespace, scripts_dir: Path, run_dir: Path) -> List[str]:
    """
    Stage 05: free-boundary Grad–Shafranov equilibrium solve.
    """
    script = scripts_dir / STAGE_TO_SCRIPT["05"]
    device_for_step = _archived_or_given(run_dir, args.device)
    target_for_step = _archived_or_given(run_dir, args.target)
    solver_for_step = _archived_or_given(run_dir, args.solver)

    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--device", str(device_for_step),
        "--target", str(target_for_step),
        "--solver", str(solver_for_step),
        "--log-level", args.log_level,
    ]

    # passthrough flags (optional but useful)
    if args.use_fixed_init:
        cmd.append("--use-fixed-init")
    if args.no_fields_05:
        cmd.append("--no-fields")

    return cmd


# Registry of command builders
CMD_BUILDERS: Dict[str, Callable[..., List[str]]] = {
    "00": _cmd_00,   # special: no run_dir yet
    "01": _cmd_01,
    "02": _cmd_02,
    "03": _cmd_03,
    "04": _cmd_04,
    "05": _cmd_05,
}

CMD_BUILDERS["04"] = _cmd_04

def _run_quicklook(args, scripts_dir, run_dir):
    if args.no_quicklook:
        return
    quicklook = scripts_dir / QUICKLOOK_SCRIPT
    cmd = [
        sys.executable, str(quicklook),
        "--run-dir", str(run_dir),
        "--formats", *args.quicklook_formats,
        "--dpi", str(args.quicklook_dpi),
        "--greens-max", str(args.quicklook_greens_max),
    ]

    if args.quicklook_fieldlines:
        cmd.append("--fieldlines")
        cmd.extend(["--fieldline-steps", str(args.quicklook_fieldline_steps)])
        cmd.extend(["--fieldline-ds", str(args.quicklook_fieldline_ds)])

    _run_subprocess(cmd)


# ============================================================
# MAIN WORKFLOW LOOP
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokamak_design workflow driver.")
    parser.add_argument("--device", required=True, help="baseline_device.yaml")
    parser.add_argument("--target", required=True, help="target_equilibrium.yaml")
    parser.add_argument("--solver", required=True, help="solver.yaml")
    parser.add_argument("--run-root", default="data/runs", help="run root dir")
    parser.add_argument("--prefix", default="", help="run id suffix/prefix")
    parser.add_argument("--notes", default="", help="notes stored in /meta/notes")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    parser.add_argument("--to", default="01", help="final stage to run (00–05)")
    parser.add_argument("--create-placeholders", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    # Stage-02 passthrough
    parser.add_argument("--write-psi-boundary-alias", action="store_true")

    # Stage-05 passthrough
    parser.add_argument("--use-fixed-init", action="store_true",
                        help="Stage 05: use /equilibrium/fixed/psi as initial guess if available")
    parser.add_argument("--no-fields-05", action="store_true",
                        help="Stage 05: skip computing fields/derived (equilibrium only)")

    # Quicklook options
    parser.add_argument("--no-quicklook", action="store_true")
    parser.add_argument("--quicklook-formats", nargs="+", default=["png"])
    parser.add_argument("--quicklook-dpi", type=int, default=160)
    parser.add_argument("--quicklook-greens-max", type=int, default=8)
    parser.add_argument("--quicklook-fieldlines", action="store_true",
                        help="Quicklook: also plot exemplary 3D field lines (if fields exist)")
    parser.add_argument("--quicklook-fieldline-steps", type=int, default=2500,
                        help="Quicklook: fieldline integration steps")
    parser.add_argument("--quicklook-fieldline-ds", type=float, default=0.01,
                        help="Quicklook: fieldline ds (pseudo arclength step)")


    args = parser.parse_args()

    # Validate stage target
    to_stage = args.to.zfill(2)
    if to_stage not in STAGE_TO_SCRIPT:
        raise ValueError(f"Unknown stage '{args.to}'. Known: {sorted(STAGE_TO_SCRIPT)}")

    scripts_dir = _scripts_dir()

    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Determine stages to execute
    end_idx = STAGE_ORDER.index(to_stage)
    stages_to_run = STAGE_ORDER[: end_idx + 1]

    run_dir: Optional[Path] = None

    # -----------------------------
    # Execute stages sequentially
    # -----------------------------
    for stage in stages_to_run:
        if stage == "00":
            cmd = _cmd_00(args, scripts_dir, run_root)
            _run_subprocess(cmd)
            run_dir = _detect_newest_run_dir(run_root)
            print(f"\n>>> Detected run_dir: {run_dir}")
        else:
            assert run_dir is not None
            cmd = CMD_BUILDERS[stage](args, scripts_dir, run_dir)
            _run_subprocess(cmd)

    assert run_dir is not None

    # Quicklook always runs at the end (unless disabled)
    _run_quicklook(args, scripts_dir, run_dir)

    print(f"\nWorkflow completed up to stage {to_stage}.")
    print(f"  run_dir: {run_dir}")


if __name__ == "__main__":
    main()
