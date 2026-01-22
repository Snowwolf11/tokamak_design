#!/usr/bin/env python3
"""
run_workflow.py
===============

High-level workflow driver for the *new* tokamak_design pipeline.

New Workflow (equilibrium-first)
--------------------------------
Stages (planned):
  00_init_run.py                (implemented)
  01_optimize_equilibrium.py    (implemented; uses stub eq or real GS later)
  02_fit_device_to_equilibrium.py   (planned)
  03_polish_joint_optimization.py   (planned)
  04_analytics.py                   (planned)
  09_quicklook.py               (optional; plotting)

This driver:
  • runs the requested stages in the correct order
  • automatically detects the newly created run directory after stage 00
  • can resume from an existing --run-dir (skip 00)
  • optionally runs quicklook at the end

Design Philosophy
-----------------
• Each stage script remains fully usable on its own.
• This file contains no physics or numerics.
• Stage-specific differences are isolated in small command-builder functions.
• The main loop is generic and won't need refactoring when new stages are added.

Usage
-----
Create a new run and optimize equilibrium:

    python scripts/run_workflow.py \
        --prefix baseline \
        --notes "some notes" \
        --to 01 \
        --copy-inputs \
        --generate-report \
        --quicklook \
        --fieldlines

Resume an existing run (skip 00) and rerun stage 01:

    python scripts/run_workflow.py \
        --run-dir data/runs/2026-01-20T120000_baseline \
        --to 01 \
        --overwrite

Run quicklook at the end:

    python scripts/run_workflow.py ... --quicklook

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

# Order in which stages must be executed (dependencies)
STAGE_ORDER = ["00", "01", "02", "03", "04", "09"]

# Mapping from stage code to script filename in scripts/
STAGE_TO_SCRIPT = {
    "00": "00_init_run.py",
    "01": "01_optimize_equilibrium.py",
    # Planned (not necessarily implemented yet)
    "02": "02_fit_device_to_equilibrium.py",
    "03": "03_polish_joint_optimization.py",
    "04": "04_analytics.py",
    "09": "09_quicklook.py",
}

IMPLEMENTED = {"00", "01", "09"}  # update as you implement more


# ============================================================
# GENERIC UTILITIES
# ============================================================

def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_subprocess(cmd: List[str]) -> None:
    print("\n>>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _detect_newest_run_dir(run_root: Path) -> Path:
    run_dirs = [p for p in run_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {run_root}.")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# STAGE-SPECIFIC COMMAND BUILDERS
# ============================================================

def _cmd_00(args: argparse.Namespace, scripts_dir: Path, run_root: Path) -> List[str]:
    """
    Stage 00: initialize run + results.h5, auto-discover configs, store configs in HDF5.
    """
    script = scripts_dir / STAGE_TO_SCRIPT["00"]

    cmd = [
        sys.executable, str(script),
        "--run-root", str(run_root),
        "--prefix", args.prefix,
        "--notes", args.notes,
        "--log-level", args.log_level,
    ]

    # Pass through new-00 options
    if args.base_dir:
        cmd += ["--base-dir", args.base_dir]
    if args.config_dir_name:
        cmd += ["--config-dir-name", args.config_dir_name]

    # required stems (optional)
    if args.require:
        cmd += ["--require", *args.require]

    if args.copy_inputs:
        cmd.append("--copy-inputs")

    if args.create_placeholders:
        cmd.append("--create-placeholders")
    if args.quicklook:
        cmd.append("--quicklook")

    return cmd


def _cmd_01(args: argparse.Namespace, scripts_dir: Path, run_dir: Path) -> List[str]:
    """
    Stage 01: optimize equilibrium (fixed-boundary GS / stub eq).
    """
    script = scripts_dir / STAGE_TO_SCRIPT["01"]

    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        "--log-level", args.log_level,
    ]

    # Stage-01 passthrough
    if args.optimizer:
        cmd += ["--optimizer", args.optimizer]
    if args.max_evals and args.max_evals > 0:
        cmd += ["--max-evals", str(args.max_evals)]
    if args.seed is not None and args.seed >= 0:
        cmd += ["--seed", str(args.seed)]

    if args.overwrite:
        cmd.append("--overwrite")
    if args.snapshot_history:
        cmd.append("--snapshot-history")
    if args.generate_report:
        cmd.append("--generate-report")

    return cmd


def _cmd_09(args: argparse.Namespace, scripts_dir: Path, run_dir: Path) -> List[str]:
    """
    Stage 09: quicklook plots (optional).
    Keep this thin; add flags later as your 09 grows.
    """
    script = scripts_dir / STAGE_TO_SCRIPT["09"]
    cmd = [
        sys.executable, str(script),
        "--run-dir", str(run_dir),
        #"--log-level", args.log_level,
        "--formats", "png",
    ]

    if args.fieldlines:
        cmd.append("--fieldlines")
    return cmd


CMD_BUILDERS: Dict[str, Callable[..., List[str]]] = {
    "00": _cmd_00,
    "01": _cmd_01,
    "09": _cmd_09,
    # 02–04 will be added later
}


# ============================================================
# MAIN WORKFLOW LOOP
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run tokamak_design workflow driver (new workflow).")

    # Either create a new run (stage 00) or resume an existing one
    parser.add_argument(
        "--run-dir",
        default="",
        help="Existing run directory to resume. If provided, stage 00 is skipped.",
    )

    # Stage 00 options
    parser.add_argument("--run-root", default="data/runs", help="Root directory where runs are created.")
    parser.add_argument("--prefix", default="", help="Optional run ID suffix/prefix.")
    parser.add_argument("--notes", default="", help="Notes stored in /meta/notes.")
    parser.add_argument("--base-dir", default="", help="Base directory containing config dir (optional).")
    parser.add_argument("--config-dir-name", default="", help="Config directory name inside base-dir (optional).")
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help=(
            "Optional list of required config stems (no extension). "
            "Example: --require equilibrium_space equilibrium_optimization device_space"
        ),
    )
    parser.add_argument("--copy-inputs", action="store_true", help="Stage 00: copy discovered YAMLs into run_dir/inputs/")
    parser.add_argument("--create-placeholders", action="store_true", help="Stage 00: create placeholder groups in HDF5")

    # Stage 01 options
    parser.add_argument("--optimizer", default="", help="Stage 01: override optimizer name (optional).")
    parser.add_argument("--max-evals", type=int, default=0, help="Stage 01: override max evaluations (0=use config).")
    parser.add_argument("--seed", type=int, default=-1, help="Stage 01: RNG seed (-1=use config/default).")
    parser.add_argument("--snapshot-history", action="store_true", help="Stage 01: snapshot existing stage group before overwrite.")
    parser.add_argument("--generate-report", action="store_true", help="Generate a pdf report of the stage results after the it run.")

    # Global controls
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--to", default="01", help="Final stage to run (00–04 or 09).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite stage outputs where supported.")
    parser.add_argument("--quicklook", action="store_true", help="Run stage 09_quicklook at end (if implemented).")
    parser.add_argument("--fieldlines", action="store_true", help="Also plot exemplary 3D field lines if fields exist")

    args = parser.parse_args()

    # Validate stage target
    to_stage = args.to.zfill(2)
    if to_stage not in STAGE_TO_SCRIPT:
        raise ValueError(f"Unknown stage '{args.to}'. Known: {sorted(STAGE_TO_SCRIPT)}")

    # Compute which stages to run
    end_idx = STAGE_ORDER.index(to_stage)
    stages_to_run = STAGE_ORDER[: end_idx + 1]

    scripts_dir = _scripts_dir()

    # Determine run_dir
    run_dir: Optional[Path] = None
    if args.run_dir.strip():
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"--run-dir does not exist or is not a directory: {run_dir}")

    run_root = Path(args.run_root).expanduser().resolve()
    _ensure_dir(run_root)

    # If resuming from --run-dir, skip stage 00 automatically
    if run_dir is not None and "00" in stages_to_run:
        stages_to_run = [s for s in stages_to_run if s != "00"]

    # If user asks for an unimplemented stage, fail early with a helpful message.
    for s in stages_to_run:
        if s not in IMPLEMENTED:
            raise RuntimeError(
                f"Stage {s} is not implemented in this workflow driver yet "
                f"(script would be: {STAGE_TO_SCRIPT.get(s)}). "
                f"Implemented: {sorted(IMPLEMENTED)}"
            )

    # -----------------------------
    # Execute stages sequentially
    # -----------------------------
    for stage in stages_to_run:
        if stage == "00":
            cmd = CMD_BUILDERS["00"](args, scripts_dir, run_root)
            _run_subprocess(cmd)
            run_dir = _detect_newest_run_dir(run_root)
            print(f"\n>>> Detected run_dir: {run_dir}")
            continue

        assert run_dir is not None, "run_dir must be known by now"
        cmd = CMD_BUILDERS[stage](args, scripts_dir, run_dir)
        _run_subprocess(cmd)

    assert run_dir is not None

    # Optional quicklook at end
    if args.quicklook:
        if "09" not in IMPLEMENTED:
            print("\n>>> Quicklook requested but stage 09 not implemented; skipping.")
        else:
            cmd = CMD_BUILDERS["09"](args, scripts_dir, run_dir)
            _run_subprocess(cmd)

    print(f"\nWorkflow completed up to stage {to_stage}.")
    print(f"  run_dir: {run_dir}")


if __name__ == "__main__":
    main()