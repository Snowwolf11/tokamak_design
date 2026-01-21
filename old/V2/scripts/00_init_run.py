#!/usr/bin/env python3
"""
00_init_run.py
==============

Initialize a new run directory and a fresh results.h5 file.

UPDATED BEHAVIOR (per workflow change)
--------------------------------------
1) Config discovery:
   • Auto-discover a config directory inside a given base directory.
   • Load ALL *.yaml / *.yml files found there.
   • --require is OPTIONAL:
       - If provided: verify those config stems exist (hard error if missing).
       - If not provided: accept everything found and log a warning.

2) Config storage in HDF5 (no run_dir/input folder):
   • Config files are NOT copied anywhere.
   • All config content is written into results.h5 under:
       /config/<config_file_stem>/<path_in_config>

3) run-root is OPTIONAL:
   • If not provided, default is: <base_dir>/data/runs
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import subprocess
from typing import Dict, Any

import yaml
import h5py
import numpy as np

from tokdesign.io.paths import make_run_id, create_run_dir
from tokdesign.io.logging_utils import setup_logger
from tokdesign.io.h5 import h5_ensure_group, h5_write_scalar, h5_write_array, h5_write_strings
from tokdesign.io.schema import validate_h5_structure


# ============================================================
# HELPERS: YAML + GIT + PATHS
# ============================================================

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML into a Python dict (safe loader)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict: {path}")
    return data


def _get_git_commit(project_root: Path) -> str:
    """Best-effort git commit hash for provenance."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


def _infer_project_root() -> Path:
    """Infer project root as scripts/../ (repo root)."""
    here = Path(__file__).resolve()
    return here.parent.parent


def _find_config_dir(base_dir: Path, *, preferred: str | None = None) -> Path:
    """Find a config directory inside base_dir."""
    base_dir = base_dir.expanduser().resolve()
    candidates: list[Path] = []
    if preferred:
        candidates.append(base_dir / preferred)
    else:
        candidates.extend([base_dir / "configs", base_dir / "config"])

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find a config directory inside base_dir={base_dir}. "
        f"Tried: {tried}"
    )


def _discover_yaml_files(config_dir: Path) -> list[Path]:
    """Return all yaml/yml files (non-recursive) sorted by name."""
    paths = sorted([*config_dir.glob("*.yaml"), *config_dir.glob("*.yml")])
    return [p.resolve() for p in paths]


def _stem(p: Path) -> str:
    return p.name.rsplit(".", 1)[0]


def _build_config_map(yaml_paths: list[Path]) -> Dict[str, Path]:
    """Map config file stem -> path. Error on duplicate stems."""
    out: Dict[str, Path] = {}
    dups: Dict[str, list[Path]] = {}

    for p in yaml_paths:
        s = _stem(p)
        if s in out:
            dups.setdefault(s, [out[s]]).append(p)
        else:
            out[s] = p

    if dups:
        msg_lines = ["Duplicate config basenames (stems) found:"]
        for s, paths in dups.items():
            msg_lines.append(f"  - {s}:")
            for pp in paths:
                msg_lines.append(f"      {pp}")
        raise ValueError("\n".join(msg_lines))

    return out


def _parse_required_list(tokens: list[str] | None) -> list[str]:
    """
    Accept:
      - space-separated tokens: --require a b c
      - comma-separated tokens: --require "a,b,c"
    """
    if not tokens:
        return []
    out: list[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "," in tok:
            out.extend([p.strip() for p in tok.split(",") if p.strip()])
        else:
            out.append(tok)
    return out


# ============================================================
# HELPERS: HDF5 WRITING OF ARBITRARY YAML
# ============================================================

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, bool, int, float, np.number)) or x is None


def _sanitize_key(k: str) -> str:
    """Make dict keys safe as HDF5 path segments."""
    return str(k).replace("/", "_")


def _write_yaml_node_to_h5(h5: h5py.File, base: str, node: Any) -> None:
    """
    Recursively write a YAML node into HDF5.

    Dicts -> subgroups
    Lists -> arrays/strings when homogeneous; else subgroups by index
    Scalars -> scalar datasets
    """
    base = "/" + base.strip("/")

    if isinstance(node, dict):
        h5_ensure_group(h5, base)
        for k, v in node.items():
            kk = _sanitize_key(k)
            _write_yaml_node_to_h5(h5, f"{base}/{kk}", v)
        return

    if isinstance(node, (list, tuple)):
        if len(node) == 0:
            h5_write_strings(h5, base, [], attrs={"yaml_type": "list", "empty": 1})
            return

        if all(isinstance(x, dict) for x in node):
            h5_ensure_group(h5, base)
            for i, item in enumerate(node):
                _write_yaml_node_to_h5(h5, f"{base}/{i}", item)
            return

        if all(isinstance(x, str) for x in node):
            h5_write_strings(h5, base, list(node), attrs={"yaml_type": "list[str]"})
            return

        if all(_is_scalar(x) for x in node):
            if all((x is not None) and isinstance(x, (bool, int, float, np.number)) for x in node):
                arr = np.asarray(node)
                if arr.dtype == object:
                    h5_write_strings(
                        h5, base, [repr(x) for x in node],
                        attrs={"yaml_type": "list[mixed_scalar_repr]"},
                    )
                else:
                    h5_write_array(h5, base, arr, attrs={"yaml_type": "list[numeric_or_bool]"})
                return

            h5_write_strings(
                h5,
                base,
                [("null" if x is None else str(x)) for x in node],
                attrs={"yaml_type": "list[mixed_scalar_as_str]"},
            )
            return

        h5_ensure_group(h5, base)
        for i, item in enumerate(node):
            _write_yaml_node_to_h5(h5, f"{base}/{i}", item)
        return

    # scalar
    if node is None:
        h5_write_scalar(h5, base, "null", attrs={"yaml_type": "null", "is_null": 1})
        return
    if isinstance(node, str):
        h5_write_scalar(h5, base, node, attrs={"yaml_type": "str"})
        return
    if isinstance(node, bool):
        h5_write_scalar(h5, base, int(node), attrs={"yaml_type": "bool"})
        return
    if _is_number(node):
        h5_write_scalar(h5, base, node, attrs={"yaml_type": "number"})
        return

    h5_write_scalar(h5, base, repr(node), attrs={"yaml_type": "repr"})


def _pick_schema_version(configs: Dict[str, Dict[str, Any]]) -> str:
    for name in sorted(configs.keys()):
        v = (configs[name].get("meta", {}) or {}).get("schema_version")
        if v:
            return str(v)
    return "0.1"


def _pick_meta_name(configs: Dict[str, Dict[str, Any]]) -> str:
    preferred_order = ["baseline_device", "device", "target_equilibrium", "target", "solver"]
    for key in preferred_order:
        if key in configs:
            name = (configs[key].get("meta", {}) or {}).get("name")
            if name:
                return str(name).strip()
    for n in sorted(configs.keys()):
        name = (configs[n].get("meta", {}) or {}).get("name")
        if name:
            return str(name).strip()
    return ""


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a tokamak_design run directory and results.h5 (auto-config discovery)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="",
        help="Base directory that contains the config directory (default: inferred repo root)",
    )
    parser.add_argument(
        "--config-dir-name",
        type=str,
        default="",
        help="Config directory name inside base-dir (e.g. 'configs'). If empty, tries 'configs' then 'config'.",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        help=(
            "Optional required config stems (without .yaml). "
            "Example: --require baseline_device target_equilibrium solver "
            "Also accepts: --require 'baseline_device,target_equilibrium,solver'"
        ),
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="",
        help="Optional run root. If omitted: <base_dir>/data/runs",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional run ID suffix (e.g. 'baseline', 'scan001')",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-form notes stored in /meta/notes",
    )
    parser.add_argument(
        "--create-placeholders",
        action="store_true",
        help="Create placeholder groups /grid, /device, /target in results.h5",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    project_root = _infer_project_root()
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir.strip() else project_root

    config_dir = _find_config_dir(base_dir, preferred=args.config_dir_name.strip() or None)
    yaml_paths = _discover_yaml_files(config_dir)
    if not yaml_paths:
        raise FileNotFoundError(f"No .yaml/.yml files found in config directory: {config_dir}")

    config_map = _build_config_map(yaml_paths)

    required = _parse_required_list(args.require)
    if required:
        missing = [r for r in required if r not in config_map]
        if missing:
            found = ", ".join(sorted(config_map.keys()))
            req = ", ".join(required)
            miss = ", ".join(missing)
            raise FileNotFoundError(
                "Missing required config files in config directory.\n"
                f"  base_dir:    {base_dir}\n"
                f"  config_dir:  {config_dir}\n"
                f"  required:    {req}\n"
                f"  missing:     {miss}\n"
                f"  found_stems: {found}\n"
                "Note: pass stems without extension."
            )

    # Load ALL discovered configs
    configs: Dict[str, Dict[str, Any]] = {}
    for stem, path in sorted(config_map.items()):
        try:
            configs[stem] = _load_yaml(path)
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML: {path}\n{e}") from e

    schema_version = _pick_schema_version(configs)
    meta_name = _pick_meta_name(configs)
    prefix = args.prefix.strip() or meta_name.strip() or ""
    run_id = make_run_id(prefix=prefix)

    # run-root default: <base_dir>/data/runs
    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root.strip()
        else (base_dir / "data" / "runs").resolve()
    )

    paths = create_run_dir(run_root, run_id)

    logger = setup_logger(paths["log_path"], level=args.log_level)
    logger.info("Initialized run directory: %s", str(paths["run_dir"]))
    logger.info("Run ID: %s", run_id)
    logger.info("Base dir: %s", str(base_dir))
    logger.info("Config dir: %s", str(config_dir))
    logger.info("Discovered %d YAML files", len(configs))

    if required:
        logger.info("Required stems: %s", ", ".join(required))
    else:
        logger.warning(
            "No required configs were provided via --require. "
            "Proceeding with all discovered configs: %s",
            ", ".join(sorted(configs.keys())),
        )

    git_commit = _get_git_commit(project_root)
    created_utc = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

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
        if required:
            h5_write_strings(h5, "/meta/required_config_stems", required)

        h5_ensure_group(h5, "/history")

        # configs (stored directly under /config, per new convention)
        h5_ensure_group(h5, "/config")
        for stem, cfg in sorted(configs.items()):
            base = f"/config/{stem}"
            h5_ensure_group(h5, base)
            # provenance
            h5_write_scalar(h5, f"{base}/_source_filename", str(config_map[stem].name))
            h5_write_scalar(h5, f"{base}/_source_path", str(config_map[stem]))
            _write_yaml_node_to_h5(h5, base, cfg)

        if args.create_placeholders:
            for g in ["/grid", "/device", "/target", "/equilibrium", "/fields", "/derived", "/analysis", "/optimization"]:
                h5_ensure_group(h5, g)

    logger.info("Created HDF5 file: %s", str(results_path))

    validate_h5_structure(results_path, schema_version=str(schema_version), stage="init")
    logger.info("Schema validation passed for stage='init'")

    print("\nRun created successfully.")
    print(f"  run_dir:      {paths['run_dir']}")
    print(f"  results.h5:   {paths['results_path']}")
    print(f"  figures/:     {paths['figures_dir']}")
    print(f"  run.log:      {paths['log_path']}")
    print(f"  schema:       {schema_version}")
    print(f"  git_commit:   {git_commit}")
    print(f"  config_dir:   {config_dir}")
    print(f"  run_root:     {run_root}")
    print(f"  configs:      {', '.join(sorted(configs.keys()))}")
    if args.notes:
        print(f"  notes:        {args.notes}")
    print("")


if __name__ == "__main__":
    main()
