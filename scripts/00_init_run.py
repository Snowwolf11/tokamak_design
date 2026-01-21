#!/usr/bin/env python3
"""
00_init_run.py
============== test

Initialize a new "run" directory and a fresh results.h5 file.

NEW BEHAVIOR (workflow change)
------------------------------
1) Config discovery:
   • Instead of passing individual YAML paths, this script auto-discovers a config
     directory inside a given base directory (default: inferred repo root).
   • It loads ALL *.yaml / *.yml files found there.
   • It then checks that a required list (given via CLI) is present (if provided).
   • Errors are explicit: missing files, duplicate basenames, YAML parse failures, etc.

2) Config provenance storage:
   • Each config is written into results.h5 under:
       /input/<config_file_stem>/<path_in_config>
     Example:
       configs/baseline_device.yaml with key device.coils.radii ->
       /input/baseline_device/device/coils/radii

3) Optional config copy:
   • With --copy-inputs, the discovered YAMLs are also copied into:
       data/runs/<run_id>/inputs/
   • This is a convenience for humans; results.h5 remains the source of truth.

Everything else (run dir creation, /meta/*, schema validation) remains similar.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, Any

import h5py

from tokdesign.io import config as cfgio
from tokdesign.io import paths as runpaths
from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.h5 import (
    h5_ensure_group,
    h5_write_scalar,
    h5_write_strings,
    h5_write_yaml_tree,   # <-- you add this to h5.py
)
from tokdesign.io.schema import validate_h5_structure


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Initialize a tokamak_design run directory and results.h5 (auto-config discovery)"
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="Base directory that contains the config directory (default: inferred repo root)",
    )
    p.add_argument(
        "--config-dir-name",
        type=str,
        default="",
        help="Config directory name inside base-dir (e.g. 'configs'). If empty, tries 'configs' then 'config'.",
    )
    p.add_argument(
        "--require",
        nargs="*",
        default=None,
        help=(
            "Required config file stems (without .yaml). "
            "Example: --require baseline_device target_equilibrium solver "
            "Also accepts a single comma-separated token."
        ),
    )
    p.add_argument(
        "--run-root",
        type=str,
        default="data/runs",
        help="Root directory where runs are created (default: data/runs)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional run ID suffix (e.g. 'baseline', 'scan001')",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-form notes stored in /meta/notes",
    )
    p.add_argument(
        "--copy-inputs",
        action="store_true",
        help=(
            "Copy all discovered config YAMLs into <run_dir>/inputs/. "
            "results.h5 remains the source of truth; this is for human convenience."
        ),
    )
    p.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Create placeholder groups /grid, /device, /target in results.h5",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- locate base + config dir ---
    project_root = cfgio.infer_project_root(__file__)
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir.strip() else project_root

    preferred = args.config_dir_name.strip() or None
    config_dir = cfgio.find_config_dir(base_dir, preferred=preferred)

    yaml_paths = cfgio.discover_yaml_files(config_dir, recursive=False)
    if not yaml_paths:
        raise FileNotFoundError(f"No .yaml/.yml files found in config directory: {config_dir}")

    config_map = cfgio.build_config_map(yaml_paths)

    # --- required list (optional) ---
    if args.require is None:
        print(
            "WARNING: --require not provided. Continuing with all YAMLs found in the config directory.\n"
            "Example: --require device_space equilibrium_space equilibrium_optimization"
        )
        required: list[str] = []
    else:
        required = cfgio.normalize_required_list(args.require)
        if not required:
            raise SystemExit(
                "ERROR: --require was provided but empty. "
                "Example: --require baseline_device target_equilibrium solver"
            )
        cfgio.check_required_stems(config_map, required)

    # --- load YAML configs ---
    configs: Dict[str, Dict[str, Any]] = {}
    for stem, path in sorted(config_map.items()):
        try:
            configs[stem] = cfgio.load_yaml(path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML: {path}\n{e}") from e

    schema_version = cfgio.pick_schema_version(configs)
    meta_name = cfgio.pick_meta_name(configs)
    prefix = args.prefix.strip() or meta_name.strip() or ""

    # --- create run dir ---
    run_id = runpaths.make_run_id(prefix=prefix)
    run_root = Path(args.run_root).expanduser().resolve()
    paths = runpaths.create_run_dir(run_root, run_id)

    # --- logging ---
    logger = setup_logger(paths["log_path"], level=args.log_level)
    logger.info("Initialized run directory: %s", str(paths["run_dir"]))
    logger.info("Run ID: %s", run_id)
    logger.info("Base dir: %s", str(base_dir))
    logger.info("Config dir: %s", str(config_dir))
    logger.info("Discovered %d YAML files: %s", len(configs), ", ".join(sorted(configs.keys())))

    if args.require is None:
        logger.warning("No required config stems provided; proceeding with discovered set.")
    else:
        logger.info("Required stems: %s", ", ".join(required))

    # --- optional: copy YAMLs into run inputs/ using paths.copy_inputs ---
    copied_paths: list[Path] = []
    inputs_dir = Path(paths["run_dir"]) / "inputs"
    if args.copy_inputs:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        # paths.copy_inputs expects list[Path]
        runpaths.copy_inputs(list(config_map.values()), inputs_dir)
        # derive destination list for meta (copy_inputs currently doesn't return)
        copied_paths = [inputs_dir / p.name for p in config_map.values()]
        logger.info("Copied %d config files into: %s", len(copied_paths), str(inputs_dir))

    # provenance
    git_commit = cfgio.get_git_commit(project_root)
    created_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- write results.h5 ---
    results_path = paths["results_path"]
    with h5py.File(results_path, "w") as h5:
        # meta
        h5_ensure_group(h5, "/meta")
        h5_write_scalar(h5, "/meta/schema_version", str(schema_version))
        h5_write_scalar(h5, "/meta/created_utc", created_utc)
        h5_write_scalar(h5, "/meta/run_id", run_id)
        h5_write_scalar(h5, "/meta/git_commit", git_commit)
        h5_write_scalar(h5, "/meta/notes", str(args.notes))
        h5_write_scalar(h5, "/meta/base_dir", str(base_dir))
        h5_write_scalar(h5, "/meta/config_dir", str(config_dir))
        h5_write_strings(h5, "/meta/config_stems", sorted(configs.keys()))
        h5_write_scalar(h5, "/meta/copied_inputs_enabled", int(bool(args.copy_inputs)))
        if args.copy_inputs:
            h5_write_strings(h5, "/meta/copied_input_filenames", [p.name for p in copied_paths])

        h5_ensure_group(h5, "/history")

        # input configs
        h5_ensure_group(h5, "/input")
        for stem, cfg in sorted(configs.items()):
            base = f"/input/{stem}"
            h5_ensure_group(h5, base)
            # provenance for each config
            h5_write_scalar(h5, f"{base}/_source_filename", str(config_map[stem].name))
            h5_write_scalar(h5, f"{base}/_source_path", str(config_map[stem]))
            # recursive YAML -> HDF5
            h5_write_yaml_tree(h5, base, cfg)

        # Optionally create placeholder groups (helps browsing early)
        if args.create_placeholders:
            for g in [
                "/grid",
                "/device",
                "/target",
                "/equilibrium",
                "/fields",
                "/derived",
                "/analysis",
                "/optimization",
            ]:
                h5_ensure_group(h5, g)

    logger.info("Created HDF5 file: %s", str(results_path))

    # validate init schema (ensures /meta/* exists)
    validate_h5_structure(results_path, schema_version=str(schema_version), stage="init")
    logger.info("Schema validation passed for stage='init'")

    # CLI handoff
    print("\nRun created successfully.")
    print(f"  run_dir:      {paths['run_dir']}")
    print(f"  results.h5:   {paths['results_path']}")
    print(f"  figures/:     {paths['figures_dir']}")
    print(f"  run.log:      {paths['log_path']}")
    print(f"  schema:       {schema_version}")
    print(f"  git_commit:   {git_commit}")
    print(f"  config_dir:   {config_dir}")
    print(f"  configs:      {', '.join(sorted(configs.keys()))}")
    if args.copy_inputs:
        print(f"  inputs/:      {inputs_dir}")
    if args.notes:
        print(f"  notes:        {args.notes}")
    print("")


if __name__ == "__main__":
    main()
