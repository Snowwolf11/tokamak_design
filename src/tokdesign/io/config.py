# src/tokdesign/io/config.py
"""
tokdesign.io.config
===================

Config discovery + YAML loading helpers used by stage scripts (especially 00_init_run).

What belongs here
-----------------
- Locate a config directory inside a base directory
- Discover YAML files
- Load YAML safely and normalize structures
- Build a stable stem->path mapping (with duplicate checks)
- Parse required-stems CLI inputs
- Simple provenance helpers (git commit)
- Small conventions (pick schema version / meta name)

What does NOT belong here
-------------------------
- HDF5 writing/reading (that's tokdesign.io.h5)
- Run directory creation/layout (that's tokdesign.io.paths)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
import subprocess

import yaml


# -----------------------------------------------------------------------------
# Project / repo helpers
# -----------------------------------------------------------------------------

def infer_project_root(script_file: str | Path) -> Path:
    """
    Infer the project root as the parent of the directory containing the script.

    Typical repo layout:
      <repo>/
        scripts/00_init_run.py

    infer_project_root(".../scripts/00_init_run.py") -> <repo>/
    """
    p = Path(script_file).expanduser().resolve()
    return p.parent.parent


def get_git_commit(project_root: Path) -> str:
    """
    Best-effort git commit hash for provenance.
    Returns "UNKNOWN" if not a git repo or git isn't available.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"


# -----------------------------------------------------------------------------
# Config directory discovery
# -----------------------------------------------------------------------------

def find_config_dir(base_dir: Path, preferred: Optional[str] = None) -> Path:
    """
    Find a config directory inside base_dir.

    Search order:
      - if preferred is provided: base_dir/<preferred>
      - otherwise: base_dir/configs, then base_dir/config

    Raises:
      FileNotFoundError if none exist.
    """
    base_dir = base_dir.expanduser().resolve()

    candidates: List[Path] = []
    if preferred:
        candidates.append(base_dir / preferred)
    else:
        candidates.extend([base_dir / "configs", base_dir / "config"])

    for c in candidates:
        if c.exists() and c.is_dir():
            return c

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find a config directory inside base_dir={base_dir}. Tried: {tried}"
    )


def discover_yaml_files(config_dir: Path, recursive: bool = False) -> List[Path]:
    """
    Return a sorted list of .yaml/.yml files in config_dir.

    By default, search is non-recursive to match your current workflow.
    """
    config_dir = config_dir.expanduser().resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
    if not config_dir.is_dir():
        raise NotADirectoryError(f"Config path is not a directory: {config_dir}")

    if recursive:
        paths = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))
    else:
        paths = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

    return sorted([p.resolve() for p in paths], key=lambda p: p.name.lower())


def stem_no_ext(p: Path) -> str:
    """
    Return filename stem without the last extension.
    Example: 'baseline_device.yaml' -> 'baseline_device'
    """
    return p.name.rsplit(".", 1)[0]


def build_config_map(yaml_paths: Iterable[Path]) -> Dict[str, Path]:
    """
    Build mapping: config_stem -> path.

    Rules:
      - stem is filename without extension (.yaml/.yml)
      - duplicate stems are an error (explicit and loud)
    """
    out: Dict[str, Path] = {}
    dups: Dict[str, List[Path]] = {}

    for p in yaml_paths:
        p = Path(p).expanduser().resolve()
        s = stem_no_ext(p)
        if s in out:
            dups.setdefault(s, [out[s]]).append(p)
        else:
            out[s] = p

    if dups:
        lines = ["Duplicate config stems found (stems must be unique):"]
        for s, paths in sorted(dups.items()):
            lines.append(f"  - {s}:")
            for pp in paths:
                lines.append(f"      {pp}")
        raise ValueError("\n".join(lines))

    return out


# -----------------------------------------------------------------------------
# YAML loading + normalization
# -----------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML from path using safe loader.

    Normalization:
      - empty YAML -> {}
      - top-level must be a dict (mapping); otherwise error
    """
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML must be a mapping/dict: {path}")

    return data


# -----------------------------------------------------------------------------
# CLI helpers
# -----------------------------------------------------------------------------

def parse_required_list_token(raw: str) -> List[str]:
    """
    Parse one token from a required-list CLI argument.

    Supports:
      - plain tokens: "equilibrium_space"
      - comma-separated: "equilibrium_space,device_space"
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
        return [p for p in parts if p]
    return [raw]


def normalize_required_list(tokens: Optional[Iterable[str]]) -> List[str]:
    """
    Normalize a sequence of tokens into a flat list of required stems.
    Returns [] if tokens is None.
    """
    if tokens is None:
        return []
    required: List[str] = []
    for tok in tokens:
        required.extend(parse_required_list_token(tok))
    return [r.strip() for r in required if r.strip()]


def check_required_stems(config_map: Mapping[str, Path], required: Iterable[str]) -> None:
    """
    Raise FileNotFoundError with a helpful message if required stems are missing.
    """
    req = [r.strip() for r in required if r.strip()]
    if not req:
        return

    missing = [r for r in req if r not in config_map]
    if not missing:
        return

    found = ", ".join(sorted(config_map.keys()))
    miss = ", ".join(missing)
    reqs = ", ".join(req)

    raise FileNotFoundError(
        "Missing required config files in config directory.\n"
        f"  required:    {reqs}\n"
        f"  missing:     {miss}\n"
        f"  found_stems: {found}\n"
        "Note: pass stems without extension. For 'baseline_device.yaml', stem is 'baseline_device'."
    )


# -----------------------------------------------------------------------------
# Small conventions: schema version + meta name
# -----------------------------------------------------------------------------

def pick_schema_version(configs: Mapping[str, Mapping[str, Any]], default: str = "0.1") -> str:
    """
    Prefer the first 'meta.schema_version' found (stable order by config stem), else default.
    """
    for name in sorted(configs.keys()):
        cfg = configs[name]
        meta = cfg.get("meta") if isinstance(cfg, dict) else None
        if isinstance(meta, dict) and meta.get("schema_version") is not None:
            return str(meta["schema_version"])
    return str(default)


def pick_meta_name(
    configs: Mapping[str, Mapping[str, Any]],
    preferred_order: Optional[List[str]] = None,
) -> str:
    """
    Try to find a human-friendly name (meta.name) in a preferred config first.

    preferred_order default:
      ['device_space', 'equilibrium_space', 'equilibrium_optimization']
    """
    if preferred_order is None:
        preferred_order = ["device_space", "equilibrium_space", "equilibrium_optimization"]

    for key in preferred_order:
        if key in configs:
            meta = (configs[key].get("meta") if isinstance(configs[key], dict) else None) or {}
            if isinstance(meta, dict) and meta.get("name"):
                return str(meta["name"]).strip()

    for name in sorted(configs.keys()):
        cfg = configs[name]
        meta = cfg.get("meta") if isinstance(cfg, dict) else None
        if isinstance(meta, dict) and meta.get("name"):
            return str(meta["name"]).strip()

    return ""
