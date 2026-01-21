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
# Updated schema: split equilibria into /equilibrium/fixed/* and /equilibrium/free/*
# and move fields/derived under those namespaces as well.

from typing import Dict, List
_SCHEMA_V01 = {
    # --------------------------------------------------------
    # After 00_init_run.py
    # --------------------------------------------------------
    "init": [
        # ---- run metadata
        "/meta",
        "/meta/schema_version",
        "/meta/created_utc",
        "/meta/run_id",
        "/meta/git_commit",
        "/meta/notes",

        # history of overwritten datasets (future-proof)
        "/history",                        # group

        # ---- YAML snapshots (exact YAML structure below these groups)
        "/input",
        "/input/device_space",
        "/input/equilibrium_space",
        "/input/equilibrium_optimization",  # stage1_optimization.yaml (objective+constraints+metrics)
    ],

    # --------------------------------------------------------
    # After 01_optimize_equilibrium.py  (fixed-boundary)
    # --------------------------------------------------------
    "stage01_fixed": [
        # ====================================================
        # Stage 01 root
        # ====================================================
        "/stage01_fixed",

        # ---------------- Grid used for all fixed-boundary solves
        "/stage01_fixed/grid",
        "/stage01_fixed/grid/R",            # 1D (nR,)
        "/stage01_fixed/grid/Z",            # 1D (nZ,)
        "/stage01_fixed/grid/RR",           # 2D (nZ,nR) mesh
        "/stage01_fixed/grid/ZZ",           # 2D (nZ,nR) mesh
        "/stage01_fixed/grid/dR",           # scalar
        "/stage01_fixed/grid/dZ",           # scalar

        # ---------------- Objective/constraints overview (human-readable + links to YAML)
        "/stage01_fixed/problem",
        "/stage01_fixed/problem/summary",   # string dataset (short human readable)
        "/stage01_fixed/problem/active_controls",    # table-like dataset or group
        "/stage01_fixed/problem/active_metrics",     # string array
        "/stage01_fixed/problem/active_constraints", # string array
        "/stage01_fixed/problem/active_terms",       # string array

        # ====================================================
        # Optimization trace (history over candidates/iterations)
        # ====================================================
        "/stage01_fixed/trace",
        "/stage01_fixed/trace/n_eval",              # scalar int
        "/stage01_fixed/trace/x",                   # 2D (N, n_active) float
        "/stage01_fixed/trace/x_names",             # 1D (n_active,) strings
        "/stage01_fixed/trace/x_bounds_lo",         # 1D (n_active,)
        "/stage01_fixed/trace/x_bounds_hi",         # 1D (n_active,)
        "/stage01_fixed/trace/x_init",              # 1D (n_active,)

        # feasibility / failure bookkeeping
        "/stage01_fixed/trace/feasible",            # 1D (N,) bool/int8
        "/stage01_fixed/trace/fail_code",           # 1D (N,) int (0 ok; >0 reason)
        "/stage01_fixed/trace/fail_reason",         # 1D (N,) short string (optional)

        # objective values
        "/stage01_fixed/trace/objective_total",     # 1D (N,) float
        "/stage01_fixed/trace/objective_blocks",    # group of 1D arrays by block name (optional)
        "/stage01_fixed/trace/objective_terms",     # group of 1D arrays by term name

        # constraints (store margins so >=0 means satisfied OR store signed residuals; pick one convention!)
        "/stage01_fixed/trace/constraints",
        "/stage01_fixed/trace/constraints/names",   # 1D string array
        "/stage01_fixed/trace/constraints/margins", # 2D (N, n_constraints) float
        "/stage01_fixed/trace/constraints/ok",      # 2D (N, n_constraints) bool/int8

        # core scalar metrics (keep manageable)
        "/stage01_fixed/trace/metrics",             # group
        # required scalars you listed + some essentials:
        "/stage01_fixed/trace/metrics/I_t",
        "/stage01_fixed/trace/metrics/B0",
        "/stage01_fixed/trace/metrics/volume",
        "/stage01_fixed/trace/metrics/poloidal_flux",
        "/stage01_fixed/trace/metrics/stored_energy",
        "/stage01_fixed/trace/metrics/aspect_ratio",
        "/stage01_fixed/trace/metrics/beta",
        "/stage01_fixed/trace/metrics/beta_p",
        "/stage01_fixed/trace/metrics/beta_N",
        "/stage01_fixed/trace/metrics/li",
        "/stage01_fixed/trace/metrics/kappa",
        "/stage01_fixed/trace/metrics/delta",
        "/stage01_fixed/trace/metrics/shafranov_shift",

        "/stage01_fixed/trace/metrics/q0",
        "/stage01_fixed/trace/metrics/q95",
        "/stage01_fixed/trace/metrics/q_min",
        "/stage01_fixed/trace/metrics/rho_qmin",
        "/stage01_fixed/trace/metrics/low_q_volume_fraction",
        "/stage01_fixed/trace/metrics/q_monotonicity_violation",
        "/stage01_fixed/trace/metrics/q_rational_proximity",
        "/stage01_fixed/trace/metrics/q_smoothness",

        "/stage01_fixed/trace/metrics/s_edge_mean",
        "/stage01_fixed/trace/metrics/s_edge_min",
        "/stage01_fixed/trace/metrics/s_min",
        "/stage01_fixed/trace/metrics/s_max",
        "/stage01_fixed/trace/metrics/negative_shear_extent",
        "/stage01_fixed/trace/metrics/shear_smoothness",

        "/stage01_fixed/trace/metrics/alpha_edge_mean",
        "/stage01_fixed/trace/metrics/alpha_edge_p95",
        "/stage01_fixed/trace/metrics/alpha_edge_integral",
        "/stage01_fixed/trace/metrics/s_alpha_margin_min",
        "/stage01_fixed/trace/metrics/s_alpha_negative_margin_integral",

        "/stage01_fixed/trace/metrics/p_peaking_factor",
        "/stage01_fixed/trace/metrics/dpdrho_max",
        "/stage01_fixed/trace/metrics/edge_pressure_gradient_integral",
        "/stage01_fixed/trace/metrics/j_peaking_factor",
        "/stage01_fixed/trace/metrics/current_centroid_shift",

        # ----------------------------------------------------
        # Best candidate (full data)
        # ----------------------------------------------------
        "/stage01_fixed/best",
        "/stage01_fixed/best/eval_index",          # scalar int (index into trace)
        "/stage01_fixed/best/x",                   # 1D (n_active,)
        #"/stage01_fixed/best/x_full",              # optional: full dict serialized / table
        "/stage01_fixed/best/objective_total",     # scalar
        "/stage01_fixed/best/constraints_margins", # 1D (n_constraints,)
        "/stage01_fixed/best/metrics",             # group (same names as trace scalars)

        # ---- equilibrium fields for best
        "/stage01_fixed/best/equilibrium",
        "/stage01_fixed/best/equilibrium/psi",          # 2D (nZ,nR)
        "/stage01_fixed/best/equilibrium/psi_axis",     # scalar
        "/stage01_fixed/best/equilibrium/psi_lcfs",     # scalar (usually 1.0)
        "/stage01_fixed/best/equilibrium/axis_R",       # scalar
        "/stage01_fixed/best/equilibrium/axis_Z",       # scalar

        "/stage01_fixed/best/equilibrium/lcfs",         # group
        "/stage01_fixed/best/equilibrium/lcfs/R",       # 1D (n_lcfs,)
        "/stage01_fixed/best/equilibrium/lcfs/Z",       # 1D (n_lcfs,)

        "/stage01_fixed/best/equilibrium/plasma_mask",  # 2D bool/int8 (nZ,nR)
        "/stage01_fixed/best/equilibrium/j_phi",        # 2D (nZ,nR)

        "/stage01_fixed/best/equilibrium/fields",       # group
        "/stage01_fixed/best/equilibrium/fields/BR",    # 2D (nZ,nR)
        "/stage01_fixed/best/equilibrium/fields/BZ",    # 2D (nZ,nR)
        "/stage01_fixed/best/equilibrium/fields/Bphi",  # 2D (nZ,nR)

        # ---- profiles for best (on a 1D psi/rho grid)
        "/stage01_fixed/best/profiles",
        "/stage01_fixed/best/profiles/psi_bar",         # 1D (n_psi,)
        "/stage01_fixed/best/profiles/rho",             # 1D (n_psi,)
        "/stage01_fixed/best/profiles/p",               # 1D (n_psi,)
        "/stage01_fixed/best/profiles/F",               # 1D (n_psi,)
        "/stage01_fixed/best/profiles/q",               # 1D (n_psi,)
        "/stage01_fixed/best/profiles/s",               # 1D (n_psi,) shear
        "/stage01_fixed/best/profiles/alpha",           # 1D (n_psi,) ballooning proxy

        # optional extras (helpful later)
        "/stage01_fixed/best/diagnostics",
        "/stage01_fixed/best/diagnostics/gs_iterations", # scalar int
        "/stage01_fixed/best/diagnostics/residual_norm", # scalar
    ],

    # --------------------------------------------------------
    # Final “complete” stage
    # --------------------------------------------------------
    "complete": [
        "/meta",
        "/meta/schema_version",
        "/meta/created_utc",
        "/meta/run_id",
        "/meta/git_commit",
        "/meta/notes",

        "/history",                     

        "/input",
        "/input/device_space",
        "/input/equilibrium_space",
        "/input/equilibrium_optimization", 
    
        "/stage01_fixed",

        "/stage01_fixed/grid",
        "/stage01_fixed/grid/R",            # 1D (nR,)
        "/stage01_fixed/grid/Z",            # 1D (nZ,)
        "/stage01_fixed/grid/RR",           # 2D (nZ,nR) mesh
        "/stage01_fixed/grid/ZZ",           # 2D (nZ,nR) mesh
        "/stage01_fixed/grid/dR",           # scalar
        "/stage01_fixed/grid/dZ",           # scalar

        "/stage01_fixed/problem",
        "/stage01_fixed/problem/summary",   # string dataset (short human readable)
        "/stage01_fixed/problem/active_controls",    # table-like dataset or group
        "/stage01_fixed/problem/active_metrics",     # string array
        "/stage01_fixed/problem/active_constraints", # string array
        "/stage01_fixed/problem/active_terms",       # string array

    
        "/stage01_fixed/trace",
        "/stage01_fixed/trace/n_eval",              # scalar int
        "/stage01_fixed/trace/x",                   # 2D (N, n_active) float
        "/stage01_fixed/trace/x_names",             # 1D (n_active,) strings
        "/stage01_fixed/trace/x_bounds_lo",         # 1D (n_active,)
        "/stage01_fixed/trace/x_bounds_hi",         # 1D (n_active,)
        "/stage01_fixed/trace/x_init",              # 1D (n_active,)

        "/stage01_fixed/trace/feasible",            # 1D (N,) bool/int8
        "/stage01_fixed/trace/fail_code",           # 1D (N,) int (0 ok; >0 reason)
        "/stage01_fixed/trace/fail_reason",         # 1D (N,) short string (optional)

        "/stage01_fixed/trace/objective_total",     # 1D (N,) float
        "/stage01_fixed/trace/objective_blocks",    # group of 1D arrays by block name (optional)
        "/stage01_fixed/trace/objective_terms",     # group of 1D arrays by term name

        "/stage01_fixed/trace/constraints",
        "/stage01_fixed/trace/constraints/names",   # 1D string array
        "/stage01_fixed/trace/constraints/margins", # 2D (N, n_constraints) float
        "/stage01_fixed/trace/constraints/ok",      # 2D (N, n_constraints) bool/int8

        "/stage01_fixed/trace/metrics",             # group
        # required scalars you listed + some essentials:
        "/stage01_fixed/trace/metrics/I_t",
        "/stage01_fixed/trace/metrics/B0",
        "/stage01_fixed/trace/metrics/volume",
        "/stage01_fixed/trace/metrics/poloidal_flux",
        "/stage01_fixed/trace/metrics/stored_energy",
        "/stage01_fixed/trace/metrics/aspect_ratio",
        "/stage01_fixed/trace/metrics/beta",
        "/stage01_fixed/trace/metrics/beta_p",
        "/stage01_fixed/trace/metrics/beta_N",
        "/stage01_fixed/trace/metrics/li",
        "/stage01_fixed/trace/metrics/kappa",
        "/stage01_fixed/trace/metrics/delta",
        "/stage01_fixed/trace/metrics/shafranov_shift",

        "/stage01_fixed/trace/metrics/q0",
        "/stage01_fixed/trace/metrics/q95",
        "/stage01_fixed/trace/metrics/q_min",
        "/stage01_fixed/trace/metrics/rho_qmin",
        "/stage01_fixed/trace/metrics/low_q_volume_fraction",
        "/stage01_fixed/trace/metrics/q_monotonicity_violation",
        "/stage01_fixed/trace/metrics/q_rational_proximity",
        "/stage01_fixed/trace/metrics/q_smoothness",

        "/stage01_fixed/trace/metrics/s_edge_mean",
        "/stage01_fixed/trace/metrics/s_edge_min",
        "/stage01_fixed/trace/metrics/s_min",
        "/stage01_fixed/trace/metrics/s_max",
        "/stage01_fixed/trace/metrics/negative_shear_extent",
        "/stage01_fixed/trace/metrics/shear_smoothness",

        "/stage01_fixed/trace/metrics/alpha_edge_mean",
        "/stage01_fixed/trace/metrics/alpha_edge_p95",
        "/stage01_fixed/trace/metrics/alpha_edge_integral",
        "/stage01_fixed/trace/metrics/s_alpha_margin_min",
        "/stage01_fixed/trace/metrics/s_alpha_negative_margin_integral",

        "/stage01_fixed/trace/metrics/p_peaking_factor",
        "/stage01_fixed/trace/metrics/dpdrho_max",
        "/stage01_fixed/trace/metrics/edge_pressure_gradient_integral",
        "/stage01_fixed/trace/metrics/j_peaking_factor",
        "/stage01_fixed/trace/metrics/current_centroid_shift",

        "/stage01_fixed/best",
        "/stage01_fixed/best/eval_index",          # scalar int (index into trace)
        "/stage01_fixed/best/x",                   # 1D (n_active,)
        "/stage01_fixed/best/x_full",              # optional: full dict serialized / table
        "/stage01_fixed/best/objective_total",     # scalar
        "/stage01_fixed/best/constraints_margins", # 1D (n_constraints,)
        "/stage01_fixed/best/metrics",             # group (same names as trace scalars)

        "/stage01_fixed/best/equilibrium",
        "/stage01_fixed/best/equilibrium/psi",          # 2D (nZ,nR)
        "/stage01_fixed/best/equilibrium/psi_axis",     # scalar
        "/stage01_fixed/best/equilibrium/psi_lcfs",     # scalar (usually 1.0)
        "/stage01_fixed/best/equilibrium/axis_R",       # scalar
        "/stage01_fixed/best/equilibrium/axis_Z",       # scalar

        "/stage01_fixed/best/equilibrium/lcfs",         # group
        "/stage01_fixed/best/equilibrium/lcfs/R",       # 1D (n_lcfs,)
        "/stage01_fixed/best/equilibrium/lcfs/Z",       # 1D (n_lcfs,)

        "/stage01_fixed/best/equilibrium/plasma_mask",  # 2D bool/int8 (nZ,nR)
        "/stage01_fixed/best/equilibrium/j_phi",        # 2D (nZ,nR)

        "/stage01_fixed/best/equilibrium/fields",       # group
        "/stage01_fixed/best/equilibrium/fields/BR",    # 2D (nZ,nR)
        "/stage01_fixed/best/equilibrium/fields/BZ",    # 2D (nZ,nR)
        "/stage01_fixed/best/equilibrium/fields/Bphi",  # 2D (nZ,nR)

        "/stage01_fixed/best/profiles",
        "/stage01_fixed/best/profiles/psi_bar",         # 1D (n_psi,)
        "/stage01_fixed/best/profiles/rho",             # 1D (n_psi,)
        "/stage01_fixed/best/profiles/p",               # 1D (n_psi,)
        "/stage01_fixed/best/profiles/F",               # 1D (n_psi,)
        "/stage01_fixed/best/profiles/q",               # 1D (n_psi,)
        "/stage01_fixed/best/profiles/s",               # 1D (n_psi,) shear
        "/stage01_fixed/best/profiles/alpha",           # 1D (n_psi,) ballooning proxy

        "/stage01_fixed/best/diagnostics",
        "/stage01_fixed/best/diagnostics/gs_iterations", # scalar int
        "/stage01_fixed/best/diagnostics/residual_norm", # scalar
       
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
