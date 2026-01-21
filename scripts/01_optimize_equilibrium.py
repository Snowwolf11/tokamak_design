#!/usr/bin/env python3
"""
01_optimize_equilibrium.py
=========================

Stage 01 orchestrator: equilibrium-first optimization (fixed-boundary GS).

This script is intentionally thin:
  • parse CLI
  • open results.h5
  • read inputs from /input/equilibrium_space and /input/equilibrium_optimization
  • build a Stage-01 "problem" object (library code)
  • run the optimizer (library code)
  • write outputs to /stage01_fixed/... (library code)
  • validate schema for stage01_fixed

All physics / math / optimization logic lives in src/tokdesign/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import h5py

from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.schema import validate_h5_structure
from tokdesign.io.h5 import (
    h5_read_input_config,          # reads /input/<name> as dict (yaml-like)
    h5_ensure_group,
    h5_write_scalar,
    h5_write_strings,
    # optional helpers you may implement later:
    # h5_delete_path_tree,
    # h5_snapshot_to_history,
)
from tokdesign.optimization import stage01_fixed
from tokdesign.stage_outputs import stage01_outputs
from tokdesign.stage_outputs import stage01_report


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 01: optimize fixed-boundary equilibrium (equilibrium-first target search)"
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory created by 00_init_run.py (contains results.h5).",
    )

    # Optimizer controls (thin passthrough; library owns details)
    p.add_argument("--optimizer", type=str, default="", help="Override optimizer name (optional).")
    p.add_argument("--max-evals", type=int, default=0, help="Override max evaluations (0=use config).")
    p.add_argument("--seed", type=int, default=-1, help="Random seed (-1=use config/default).")

    # Execution / IO policy
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite /stage01_fixed if it already exists.",
    )
    p.add_argument(
        "--snapshot-history",
        action="store_true",
        help="Before overwriting, snapshot existing /stage01_fixed into /history (if supported).",
    )
    p.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a pdf report of the stage results after the it run.",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return p.parse_args()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_run_dir(run_dir: str) -> Path:
    p = Path(run_dir).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist or is not a directory: {p}")
    return p


def _stage01_exists(h5: h5py.File) -> bool:
    return "/stage01_fixed" in h5


# ----------------------------
# Stage runner
# ----------------------------

def run_stage01(run_dir: Path, args: argparse.Namespace) -> None:
    results_path = run_dir / "results.h5"
    if not results_path.exists():
        raise FileNotFoundError(f"results.h5 not found in run dir: {results_path}")

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path, level=args.log_level)

    logger.info("========== Stage 01: optimize equilibrium (fixed-boundary) ==========")
    logger.info("run_dir: %s", str(run_dir))
    logger.info("results.h5: %s", str(results_path))

    started_utc = _utc_now_iso()

    with h5py.File(results_path, "a") as h5:
        # --- overwrite policy ---
        if _stage01_exists(h5):
            if not args.overwrite:
                raise RuntimeError(
                    "Path /stage01_fixed already exists. "
                    "Use --overwrite to overwrite it (and optionally --snapshot-history)."
                )

            # Optional snapshot support (pretend exists; implement later if desired)
            if args.snapshot_history:
                logger.info("Snapshot requested, but snapshot helper may not be implemented yet.")
                # Example future call:
                # h5_snapshot_to_history(h5, "/stage01_fixed", note="before overwrite stage01")
                # h5_delete_path_tree(h5, "/stage01_fixed")

            # Minimal behavior for now: delete + recreate OR let writer overwrite datasets.
            # If you implement h5_delete_path_tree, uncomment:
            # h5_delete_path_tree(h5, "/stage01_fixed")

        # --- read inputs (ONLY from HDF5) ---
        cfg_space = h5_read_input_config(h5, "equilibrium_space")
        cfg_opt = h5_read_input_config(h5, "equilibrium_optimization")

        # --- build problem (library) ---
        # run_context is small provenance + IO hints; library may ignore extras.
        run_context = {
            "run_dir": str(run_dir),
            "results_path": str(results_path),
            "run_id": h5.get("/meta/run_id", None)[()] if "/meta/run_id" in h5 else "",
            "git_commit": h5.get("/meta/git_commit", None)[()] if "/meta/git_commit" in h5 else "",
            "schema_version": h5.get("/meta/schema_version", None)[()] if "/meta/schema_version" in h5 else "",
        }

        # bytes -> str cleanup (common with h5py)
        for k in list(run_context.keys()):
            v = run_context[k]
            if isinstance(v, (bytes, bytearray)):
                run_context[k] = v.decode("utf-8", errors="replace")

        problem = stage01_fixed.build_problem(
            cfg_space=cfg_space,
            cfg_opt=cfg_opt,
            run_context=run_context,
        )

        # --- apply CLI overrides (thin; library decides final) ---
        opt_overrides = {
            "optimizer": args.optimizer.strip() or None,
            "max_evals": int(args.max_evals) if args.max_evals and args.max_evals > 0 else None,
            "seed": int(args.seed) if args.seed is not None and args.seed >= 0 else None,
        }

        logger.info("Optimizer overrides: %s", {k: v for k, v in opt_overrides.items() if v is not None})

        # --- run optimization (library) ---
        # Expected to return a Stage01Result-like object with:
        #   - meta (dict)
        #   - trace (arrays + names)
        #   - best (full equilibrium + profiles + metrics)
        result = stage01_fixed.run_optimization(
            problem,
            optimizer=opt_overrides["optimizer"],
            max_evals=opt_overrides["max_evals"],
            seed=opt_overrides["seed"],
        )

        finished_utc = _utc_now_iso()

        # --- write outputs (library owns schema mapping) ---
        # Preferred: library provides a single writer that fills /stage01_outputs.
        stage01_outputs.write_outputs(h5, result)

        # --- ensure a minimal meta block exists even if writer forgot ---
        h5_ensure_group(h5, "/stage01_fixed/meta")
        h5_write_scalar(h5, "/stage01_fixed/meta/stage_name", "01_optimize_equilibrium")
        h5_write_scalar(h5, "/stage01_fixed/meta/started_utc", started_utc)
        h5_write_scalar(h5, "/stage01_fixed/meta/finished_utc", finished_utc)
        if args.optimizer.strip():
            h5_write_scalar(h5, "/stage01_fixed/meta/optimizer_override", args.optimizer.strip())
        if args.max_evals and args.max_evals > 0:
            h5_write_scalar(h5, "/stage01_fixed/meta/max_evals_override", int(args.max_evals))
        if args.seed is not None and args.seed >= 0:
            h5_write_scalar(h5, "/stage01_fixed/meta/seed_override", int(args.seed))

        # Optional: store which input configs were consumed (human breadcrumb)
        h5_write_strings(
            h5,
            "/stage01_fixed/meta/inputs_used",
            ["equilibrium_space", "equilibrium_optimization"],
        )

        if args.generate_report:
            stage01_report.write_report(run_dir, results_path)
            logger.info("Stage 01 report written")

    # --- schema validation (after file is closed) ---
    # Use the same schema_version as the file already has.
    with h5py.File(results_path, "r") as h5r:
        schema_version = ""
        if "/meta/schema_version" in h5r:
            v = h5r["/meta/schema_version"][()]
            schema_version = v.decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v)

    validate_h5_structure(results_path, schema_version=schema_version, stage="stage01_fixed")
    logger.info("Schema validation passed for stage='stage01_fixed'")

    print("\nStage 01 completed successfully.")
    print(f"  run_dir:    {run_dir}")
    print(f"  results.h5: {results_path}")
    print("")


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    run_stage01(run_dir, args)


if __name__ == "__main__":
    main()
