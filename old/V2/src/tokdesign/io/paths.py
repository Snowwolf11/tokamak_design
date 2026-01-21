"""
paths.py
========

Filesystem utilities for run management.

Responsibilities
----------------
• Create unique run IDs
• Create standardized run directory structure
• Copy YAML config files into run folder for provenance
"""

from pathlib import Path
from datetime import datetime
import shutil
from typing import Optional, List


# ============================================================
# RUN ID GENERATION
# ============================================================

def make_run_id(prefix: Optional[str] = None) -> str:
    """
    Generate a timestamp-based run ID.

    Format:
        YYYY-MM-DDTHHMMSS[_prefix]

    Examples:
        2026-01-15T134522
        2026-01-15T134522_baseline
    """

    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H%M%S")

    if prefix is None or prefix == "":
        return timestamp
    else:
        safe_prefix = prefix.replace(" ", "_").replace("/", "_")
        return f"{timestamp}_{safe_prefix}"


# ============================================================
# RUN DIRECTORY CREATION
# ============================================================

def create_run_dir(base_dir: Path, run_id: str) -> dict:
    """
    Create standard run directory structure.

    Structure:
        base_dir/
            run_id/
                inputs/
                figures/
                run.log
                results.h5
                summary.csv
    """

    base_dir = Path(base_dir).expanduser().resolve()
    run_dir = base_dir / run_id

    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists:\n{run_dir}"
        )

    inputs_dir = run_dir / "inputs"
    figures_dir = run_dir / "figures"

    inputs_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)

    results_path = run_dir / "results.h5"
    summary_path = run_dir / "summary.csv"
    log_path = run_dir / "run.log"

    return {
        "run_dir": run_dir,
        "inputs_dir": inputs_dir,
        "figures_dir": figures_dir,
        "results_path": results_path,
        "summary_path": summary_path,
        "log_path": log_path,
    }


# ============================================================
# CONFIG COPY
# ============================================================

def copy_inputs(config_paths: List[Path], dest_inputs_dir: Path) -> None:
    """
    Copy YAML configuration files into run folder.
    """

    dest_inputs_dir = Path(dest_inputs_dir).expanduser().resolve()

    if not dest_inputs_dir.exists():
        raise FileNotFoundError(
            f"Destination inputs directory does not exist:\n{dest_inputs_dir}"
        )

    for cfg in config_paths:
        cfg = Path(cfg).expanduser().resolve()

        if not cfg.exists():
            raise FileNotFoundError(f"Config file not found:\n{cfg}")

        if cfg.suffix.lower() not in [".yaml", ".yml"]:
            raise ValueError(
                f"Config file does not look like YAML:\n{cfg}"
            )

        dest = dest_inputs_dir / cfg.name

        if dest.exists():
            raise FileExistsError(
                f"Config already exists in inputs folder:\n{dest}"
            )

        shutil.copy2(cfg, dest)


# ============================================================
# SMALL UTILITIES
# ============================================================

def assert_is_run_dir(run_dir: Path) -> None:
    """
    Sanity check that a directory looks like a run folder.
    """

    run_dir = Path(run_dir).expanduser().resolve()

    required = ["inputs", "figures"]

    for name in required:
        if not (run_dir / name).exists():
            raise RuntimeError(
                f"Not a valid run directory:\n{run_dir}\n"
                f"Missing: {name}"
            )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("Testing paths.py")

    base = Path("/tmp/tokamak_test_runs")
    base.mkdir(exist_ok=True)

    run_id = make_run_id("test")
    paths = create_run_dir(base, run_id)

    print("Created:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    f1 = base / "a.yaml"
    f2 = base / "b.yaml"
    f1.write_text("a: 1")
    f2.write_text("b: 2")

    copy_inputs([f1, f2], paths["inputs_dir"])

    print("Inputs copied successfully")
    print("paths.py self-test passed")
