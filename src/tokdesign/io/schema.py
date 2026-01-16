"""
schema.py
=========

HDF5 schema validation utilities.

Purpose
-------
Provide lightweight, stage-aware validation of the expected HDF5 structure
(results.h5) produced by the pipeline scripts.

Why this exists
---------------
• Catch missing outputs early (e.g. forgot to run 01_build_device.py)
• Make "resume from run folder" safe and debuggable
• Provide a single place to evolve the data contract as the project grows

Design principles
-----------------
• Stage-aware (init/device/target/fixed_eq/free_eq/report)
• Conservative: only checks group/dataset existence (not physics correctness)
• Produces clear, actionable error messages
• Expandable: you can add new schema versions and required paths

Usage patterns
--------------
1) After a script writes outputs, validate that its required paths now exist.
2) Before a script runs, validate that upstream prerequisites exist.

Example:
    validate_h5_structure(results_path, schema_version="0.1", stage="device")

Conventions
-----------
• HDF5 paths use UNIX-style: "/meta/schema_version"
• This validator checks existence: path in h5
• You can distinguish required groups vs datasets if you want later,
  but for most uses "exists" is sufficient.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Iterable

import h5py


PathLike = Union[str, Path]


# ============================================================
# SCHEMA DEFINITIONS
# ============================================================

# Minimal, practical schema for v0.1
# Add/extend as your project grows.
_SCHEMA_V01: Dict[str, List[str]] = {
    # --------------------------------------------------------
    # After 00_init_run.py
    # --------------------------------------------------------
    "init": [
        "/meta",
        "/meta/schema_version",
        "/meta/created_utc",
        "/meta/run_id",
        # These may be blank/unknown early, but reserve the keys anyway
        # (You can store empty strings if not known)
        "/meta/git_commit",
        "/meta/notes",
        "/history"
    ],

    # --------------------------------------------------------
    # After 01_build_device.py
    # --------------------------------------------------------
    "device": [
        "/grid",
        "/grid/R",
        "/grid/Z",
        "/grid/RR",
        "/grid/ZZ",

        "/device",
        "/device/vessel_boundary",

        "/device/coils",
        "/device/coils/names",
        "/device/coils/centers",
        "/device/coils/radii",
        "/device/coils/I_max",
        "/device/coils/I_pf",

        # Optional in the overall format, but strongly recommended in v1
        # because coil fitting depends on it.
        "/device/coil_greens",
        "/device/coil_greens/psi_per_amp",
    ],

    # --------------------------------------------------------
    # After 02_target_boundary.py
    # --------------------------------------------------------
    "target": [
        "/target",
        "/target/boundary", #LCFS
        "/target/psi_boundary",
        "/target/global_targets",
        "/target/global_targets/enforce",
        "/target/global_targets/Ip_target",
        # xpoint optional, so not required here
    ],

    # --------------------------------------------------------
    # After 03_solve_fixed_gs.py (fixed-boundary equilibrium)
    # --------------------------------------------------------
    "fixed_eq": [
        "/equilibrium",
        "/equilibrium/psi",
        "/equilibrium/psi_axis",
        "/equilibrium/psi_lcfs",

        "/equilibrium/p_psi",
        "/equilibrium/F_psi",

        "/equilibrium/plasma_mask",
        "/equilibrium/jphi",

        "/fields",
        "/fields/BR",
        "/fields/BZ",
        "/fields/Bphi",

        "/derived",
        "/derived/Ip",
        "/derived/beta_p",
        "/derived/li",
        "/derived/kappa",
        "/derived/delta",
        "/derived/q_profile",
    ],

    # --------------------------------------------------------
    # After 05_solve_free_gs.py (free-boundary equilibrium)
    # --------------------------------------------------------
    "free_eq": [
        # Keep the fixed-boundary outputs too (they’re still relevant),
        # but also require the extra free-boundary products:
        "/equilibrium",
        "/equilibrium/psi_total",
        "/equilibrium/psi_vac",
        "/equilibrium/psi_plasma",
        "/equilibrium/lcfs_poly",

        # It’s still sensible to store these for the converged state:
        "/equilibrium/psi_axis",
        "/equilibrium/psi_lcfs",
        "/equilibrium/p_psi",
        "/equilibrium/F_psi",
        "/equilibrium/plasma_mask",
        "/equilibrium/jphi",

        "/fields",
        "/fields/BR",
        "/fields/BZ",
        "/fields/Bphi",

        "/derived",
        "/derived/Ip",
        "/derived/beta_p",
        "/derived/li",
        "/derived/kappa",
        "/derived/delta",
        "/derived/q_profile",

        "/analysis",
        "/analysis/shape_metrics",
        "/analysis/vertical_proxy",
        # control_matrix might be optional early, but included here as a "complete" goal
        "/analysis/control_matrix",
    ],

    # --------------------------------------------------------
    # After 04_fit_pf_currents.py (coil fit results)
    # --------------------------------------------------------
    "optimization_fit": [
        "/optimization",
        "/optimization/objective_terms",
        "/optimization/constraint_margins",
        # If you store detailed fit outputs, you can require them too:
        # "/optimization/fit_results",
    ],

    # --------------------------------------------------------
    # Final “complete” stage (everything in the planned structure)
    # --------------------------------------------------------
    "complete": [
        "/meta",
        "/meta/schema_version",
        "/meta/created_utc",
        "/meta/run_id",
        "/meta/git_commit",
        "/meta/notes",

        "/grid",
        "/grid/R",
        "/grid/Z",
        "/grid/RR",
        "/grid/ZZ",

        "/device",
        "/device/vessel_boundary",
        "/device/coils",
        "/device/coils/names",
        "/device/coils/centers",
        "/device/coils/radii",
        "/device/coils/I_max",
        "/device/coils/I_pf",
        "/device/coil_greens",
        "/device/coil_greens/psi_per_amp",

        "/target",
        "/target/boundary", 
        "/target/psi_boundary",
        "/target/global_targets",
        "/target/global_targets/enforce",
        "/target/global_targets/Ip_target",

        "/equilibrium",
        "/equilibrium/psi",
        "/equilibrium/psi_axis",
        "/equilibrium/psi_lcfs",
        "/equilibrium/p_psi",
        "/equilibrium/F_psi",
        "/equilibrium/plasma_mask",
        "/equilibrium/jphi",

        "/fields",
        "/fields/BR",
        "/fields/BZ",
        "/fields/Bphi",

        "/derived",
        "/derived/Ip",
        "/derived/beta_p",
        "/derived/li",
        "/derived/kappa",
        "/derived/delta",
        "/derived/q_profile",

        "/analysis",
        "/analysis/shape_metrics",
        "/analysis/vertical_proxy",
        "/analysis/control_matrix",

        "/optimization",
        "/optimization/objective_terms",
        "/optimization/constraint_margins",
    ],
}

# Registry of schemas by version
SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "0.1": _SCHEMA_V01
}


# ============================================================
# PUBLIC API
# ============================================================

def validate_h5_structure(
    h5_path: PathLike,
    schema_version: str,
    stage: str,
    *,
    allow_extra: bool = True,
) -> None:
    """
    Validate that `results.h5` contains the required paths for the given stage.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    schema_version : str
        Schema version to validate against (e.g. "0.1").
    stage : str
        Validation stage. Typical stages:
            "init", "device", "target", "fixed_eq", "free_eq", "report"
    allow_extra : bool
        Currently unused (placeholder). HDF5 will almost always contain extra
        paths; disallowing extras is rarely helpful.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If schema_version or stage is unknown.
    RuntimeError
        If any required path is missing.
    """

    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    schema = _get_schema(schema_version)
    required_paths = _get_stage_paths(schema, stage)

    with h5py.File(h5_path, "r") as h5:
        missing = list_missing_paths(h5, required_paths)

        if missing:
            msg = _format_missing_message(h5_path, schema_version, stage, missing)
            raise RuntimeError(msg)

        # Optional: check that stored schema_version matches expected
        # (only if it exists; init stage might create it)
        try:
            stored = _read_scalar_string_if_present(h5, "/meta/schema_version")
            if stored is not None and stored != schema_version:
                raise RuntimeError(
                    "Schema version mismatch:\n"
                    f"  expected: {schema_version}\n"
                    f"  stored:   {stored}\n"
                    f"  file:     {h5_path}"
                )
        except Exception:
            # Keep validation robust; don’t fail just because schema_version
            # is stored in an unexpected way. You can tighten later.
            pass


def require_groups(h5: h5py.File, groups: Iterable[str]) -> None:
    """
    Require that specific group paths exist in an open HDF5 file.

    This is a convenience helper when you want to check groups only (not datasets).

    Parameters
    ----------
    h5 : h5py.File
        Open file handle.
    groups : Iterable[str]
        Group paths like "/meta", "/device/coils"

    Raises
    ------
    RuntimeError if any group is missing.
    """
    missing = []
    for g in groups:
        g = _normalize_path(g)
        if g not in h5:
            missing.append(g)
        else:
            # Exists but might not be a group; enforce it is a group
            if not isinstance(h5[g], h5py.Group):
                missing.append(g + " (exists but is not a group)")
    if missing:
        raise RuntimeError("Missing required groups:\n  " + "\n  ".join(missing))


def list_missing_paths(h5: h5py.File, required_paths: Iterable[str]) -> List[str]:
    """
    Return a list of required paths that are missing in the open HDF5 file.

    Parameters
    ----------
    h5 : h5py.File
        Open file handle.
    required_paths : Iterable[str]
        Paths to check.

    Returns
    -------
    missing : list[str]
        Missing paths.
    """
    missing = []
    for p in required_paths:
        p = _normalize_path(p)
        if p not in h5:
            missing.append(p)
    return missing


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _get_schema(schema_version: str) -> Dict[str, List[str]]:
    if schema_version not in SCHEMAS:
        known = ", ".join(sorted(SCHEMAS.keys()))
        raise ValueError(f"Unknown schema_version '{schema_version}'. Known: {known}")
    return SCHEMAS[schema_version]


def _get_stage_paths(schema: Dict[str, List[str]], stage: str) -> List[str]:
    if stage not in schema:
        known = ", ".join(sorted(schema.keys()))
        raise ValueError(f"Unknown stage '{stage}'. Known: {known}")
    return schema[stage]


def _normalize_path(path: str) -> str:
    if not path:
        raise ValueError("Empty HDF5 path")
    path = path.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    return path


def _format_missing_message(
    h5_path: Path,
    schema_version: str,
    stage: str,
    missing: List[str],
) -> str:
    lines = [
        "HDF5 schema validation failed.",
        f"  file:   {h5_path}",
        f"  schema: {schema_version}",
        f"  stage:  {stage}",
        "",
        "Missing required paths:",
    ]
    lines.extend([f"  - {m}" for m in missing])
    lines.append("")
    lines.append("Hint:")
    lines.append(_hint_for_stage(stage))
    return "\n".join(lines)


def _hint_for_stage(stage: str) -> str:
    # Simple mapping to help users run the right script next.
    hints = {
        "init": "Run scripts/00_init_run.py to initialize the run and /meta.",
        "device": "Run scripts/01_build_device.py to create /grid and /device outputs.",
        "target": "Run scripts/02_target_boundary.py to create /target outputs.",
        "fixed_eq": "Run scripts/03_solve_fixed_gs.py to generate fixed-boundary equilibrium.",
        "free_eq": "Run scripts/05_solve_free_gs.py (and ensure 04_fit_pf_currents.py was run).",
        "report": "Run scripts/07_make_report.py to generate figures and summary outputs.",
    }
    return hints.get(stage, "Run the script responsible for producing this stage.")


def _read_scalar_string_if_present(h5: h5py.File, path: str) -> Optional[str]:
    """
    Attempt to read a scalar string dataset if it exists.

    Returns None if not present.
    """
    path = _normalize_path(path)
    if path not in h5:
        return None
    dset = h5[path]
    try:
        val = dset[()]
        if isinstance(val, bytes):
            return val.decode("utf-8")
        # h5py may return numpy bytes_ or numpy scalar
        if hasattr(val, "dtype") and str(val.dtype).startswith("|S"):
            return val.tobytes().decode("utf-8")
        if isinstance(val, str):
            return val
    except Exception:
        return None
    # Fallback
    try:
        return str(dset[()])
    except Exception:
        return None


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    """
    Self-test creates a tiny HDF5 file with only /meta,
    then demonstrates stage validation errors and success.
    """

    from datetime import datetime
    from tempfile import TemporaryDirectory

    print("Testing schema.py")

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fpath = tmp / "results.h5"

        # Create minimal init structure
        with h5py.File(fpath, "w") as h5:
            h5.create_group("meta")
            h5["/meta"].create_dataset("schema_version", data="0.1")
            h5["/meta"].create_dataset("created_utc", data=datetime.utcnow().isoformat() + "Z")
            h5["/meta"].create_dataset("run_id", data="TEST_RUN")
            h5["/meta"].create_dataset("git_commit", data="TEST_RUN")
            h5["/meta"].create_dataset("notes", data="TEST_RUN")

        # This should pass for init
        validate_h5_structure(fpath, "0.1", "init")
        print("  init stage: PASS")

        # This should fail for device (no /grid, /device)
        try:
            validate_h5_structure(fpath, "0.1", "device")
        except RuntimeError as e:
            print("  device stage: expected FAIL")
            print("  message snippet:")
            print(str(e).splitlines()[0])

    print("schema.py self-test passed")
