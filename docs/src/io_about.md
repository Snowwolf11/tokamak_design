docs/src/io.md
===================

16.01.26

Scope
-----
This document describes the `src/.../io` subfolder (Pass A: “module contract”).
It focuses on: idea/philosophy, included modules, public API, I/O contracts,
dataflow touchpoints (especially results.h5 + run directories), dependencies,
and notable implementation details.

Folder philosophy (“io”)
------------------------
The io layer exists to make the workflow *repeatable, resumable, and debuggable*.

Core rule:
  • Physics/geometry/optimization code must NOT live here.
  • io provides “plumbing”: paths, logging, HDF5 read/write helpers, schema checks,
    and (lightweight) unit conventions.

In the overall repository dependency direction, io is at the bottom:
  io → geometry → physics → {analysis, optimization} → viz

That means:
  • Everything may depend on io.
  • io should depend on as little as possible (stdlib + h5py/numpy).

What “io” standardizes
----------------------
1) Run folder layout (provenance + artifacts)
2) Logging behavior (consistent formatting, console + file)
3) Canonical data store conventions (results.h5)
4) A stage-aware schema validator (so scripts can check prerequisites/outputs)
5) Unit conventions (currently minimal; extension point for future unit library)

Modules in this folder
----------------------
- paths.py
- logging_utils.py
- h5.py
- schema.py
- units.py


================================================================================
paths.py — Run directory and provenance utilities
================================================================================

Purpose
-------
Manage the filesystem structure for a workflow “run”:
  • create unique run IDs
  • create a standard run directory tree
  • copy YAML config files into the run for provenance (“inputs/” snapshot)
  • sanity-check whether a path looks like a run directory

Public API
----------
1) make_run_id(prefix: Optional[str] = None) -> str
   - Generates UTC timestamp-based run id:
       "YYYY-MM-DDTHHMMSS" or "YYYY-MM-DDTHHMMSS_prefix"
   - `prefix` is sanitized (spaces, slashes replaced by underscores).

2) create_run_dir(base_dir: Path, run_id: str) -> dict
   - Creates:
       run_dir/
         inputs/
         figures/
         run.log
         results.h5
         summary.csv
   - Returns a dict of paths:
       {
         "run_dir", "inputs_dir", "figures_dir",
         "results_path", "summary_path", "log_path"
       }
   - Fails fast if run_dir already exists.

3) copy_inputs(config_paths: List[Path], dest_inputs_dir: Path) -> None
   - Copies YAML files into inputs_dir using shutil.copy2 (preserves metadata).
   - Guards:
       • dest_inputs_dir must exist
       • each config must exist
       • suffix must be .yaml or .yml
       • refuses to overwrite if filename already exists in inputs/

4) assert_is_run_dir(run_dir: Path) -> None
   - Minimal sanity check: requires subfolders “inputs” and “figures”.
   - (Note: it does NOT check results.h5/run.log/summary.csv existence.)

Inputs/Outputs
--------------
Inputs:
  • base_dir, run_id, list of YAML config paths
Outputs:
  • filesystem side-effects: created directories/files
  • returned path dictionary

Dependencies
------------
Stdlib only:
  pathlib, datetime, shutil, typing

Design notes / intended usage
-----------------------------
- scripts/00_init_run.py should be the main consumer:
  • generate run_id, create run structure, copy YAML configs
  • store run_id in results.h5:/meta/run_id (via h5.py)
- Other scripts should not reinvent paths; they should accept run_dir/results_path
  and rely on the standard structure created here.

Potential improvements / risks
------------------------------
- `assert_is_run_dir()` is intentionally lightweight; if you want stronger guards,
  consider optionally checking for results.h5 existence too.
- `copy_inputs()` checks only suffix, not YAML validity. That’s fine for provenance,
  but scripts should validate config content elsewhere.


================================================================================
logging_utils.py — Central logger configuration
================================================================================

Purpose
-------
Provide a single, consistent logger setup for:
  • stdout console logs
  • a persistent per-run log file (run.log)
  • formatting and level normalization
  • preventing duplicate handlers

Public API
----------
1) setup_logger(log_path: Path, level: str = "INFO") -> logging.Logger
   - Creates/returns logger with name "tokdesign"
   - Adds StreamHandler + FileHandler
   - Uses formatter:
       "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
       datefmt = "%Y-%m-%d %H:%M:%S"
   - Creates parent directories for log file path if needed

Inputs/Outputs
--------------
Inputs:
  • log_path: where run.log should be written
  • level: string ("DEBUG", "INFO", ...)
Outputs:
  • configured logger instance
  • side-effect: writes to file during runtime

Dependencies
------------
Stdlib only:
  logging, pathlib

Important implementation note (bug to fix)
------------------------------------------
There is a logic branch:

  if logger.handlers:
      for h in logger.handlers:
          h.setLevel(logger.level)
          h.setFormatter(formatter)
      return logger

But `formatter` is defined later in the function.
If `setup_logger()` is called twice in the same process (common in notebooks,
tests, or any orchestration that imports/uses scripts more than once), this
branch will raise UnboundLocalError (formatter referenced before assignment).

Recommended fix:
  • Define `formatter` before checking `logger.handlers`, OR
  • Rebuild a formatter in that branch, OR
  • Store formatter on the logger object / module-level constant.

Design notes / intended usage
-----------------------------
- The scripts layer should call setup_logger(run_dir/"run.log") once early,
  then pass logger down or use logging.getLogger("tokdesign").
- This module should remain side-effect-free except for handler creation.


================================================================================
h5.py — Low-level HDF5 read/write utilities for results.h5
================================================================================

Purpose
-------
A small, consistent API around h5py to reduce boilerplate and enforce conventions
for the canonical workflow artifact: results.h5.

Key philosophy
--------------
- This is intentionally low-level:
    “It does not know tokamak physics; it knows how to store things reliably.”
- It standardizes:
  • path normalization
  • group creation
  • overwriting rules
  • string handling (UTF-8 variable-length)
  • attr handling for small metadata
  • basic history snapshots

Public Types
------------
- PathLike = Union[str, Path]
- Attrs = Optional[Dict[str, Any]]

Public API (write)
------------------
1) h5_ensure_group(h5: h5py.File, path: str) -> h5py.Group
   - Ensures nested group path exists; creates intermediate groups.

2) h5_write_array(h5, path, arr, attrs=None, overwrite=True,
                  compression="gzip", compression_opts=4) -> None
   - Writes numpy-compatible arrays as datasets.
   - Guards against dtype=object arrays (except for explicit string writer).
   - Compression applied if enabled and arr.size > 0.

3) h5_write_scalar(h5, path, value, attrs=None, overwrite=True) -> None
   - Writes scalar as 0-d dataset.
   - If string: stored with UTF-8 variable-length dtype.

4) h5_write_strings(h5, path, strings: list[str], attrs=None, overwrite=True) -> None
   - Writes 1D string dataset using UTF-8 variable-length dtype.

5) h5_write_dict_as_attrs(h5, group_path, d: Dict[str,Any], overwrite=True) -> None
   - Writes small dict entries into group attributes.

Public API (read)
-----------------
6) h5_read_array(h5, path) -> np.ndarray
   - Reads dataset; attempts to decode bytes to str.
   - Handles scalar bytes, fixed-length byte strings, and some object arrays of bytes.

7) h5_read_scalar(h5, path) -> Any
   - Reads scalar dataset; decodes bytes; numpy scalar -> python scalar.

8) h5_read_attrs(h5, path) -> Dict[str,Any]
   - Reads attrs from dataset or group; decodes bytes when possible.

File open convenience
---------------------
9) open_h5(path: PathLike, mode: str = "r") -> h5py.File
   - Normalizes path; returns h5py.File.

History / snapshotting (provenance)
-----------------------------------
10) h5_make_history_event_id() -> str
    - Returns sortable UTC id like "2026-01-16T210534Z".

11) h5_snapshot_paths(h5, stage: str, src_paths: list[str], event_id=None,
                      attrs=None, overwrite_event=False) -> str
    - Copies existing HDF5 paths into:
        /history/<stage>/<event_id>/...
      preserving relative structure:
        src "/device/coils/I_pf" ->
        dst "/history/<stage>/<event_id>/device/coils/I_pf"
    - Skips src paths that do not exist (snapshot “what’s there”).
    - Optionally attaches attrs to the event group.

Inputs/Outputs
--------------
Inputs:
  • open h5py.File handle (caller controls open mode)
  • paths like "/device/coils/names"
Outputs:
  • HDF5 datasets/groups written to results.h5
  • history snapshots stored under /history

Dependencies
------------
- numpy
- h5py
- stdlib: pathlib, typing, datetime

Design notes / conventions enforced
-----------------------------------
- Path normalization:
  • ensures leading slash
  • collapses duplicate slashes
  • replaces backslashes with forward slashes (Windows safety)
- Overwrite is explicit and defaults to True for datasets; safe for pipeline steps
  that want “last writer wins” semantics.
- Attributes are intended for “small metadata”; large structured data should be
  datasets or subgroups.

What it does NOT do (by design)
-------------------------------
- No schema validation (that’s schema.py)
- No physics correctness checks
- No knowledge of specific group names beyond paths passed in


================================================================================
schema.py — Stage-aware HDF5 structure validation
================================================================================

Purpose
-------
Provide a conservative “contract checker” for results.h5:
  • before a stage runs: verify prerequisites exist (previous outputs)
  • after a stage runs: verify outputs were written

Philosophy
----------
- Validation is *structural* only:
  • checks presence/absence of required paths
  • does not verify numerical ranges, shapes, or physics correctness
- Stage-aware:
  • separate required paths per stage
- Clear errors:
  • missing paths listed explicitly
  • includes a “hint” mapping stage -> which script to run next

Schema registry
---------------
- SCHEMAS is a dict of schema versions -> stage definitions.
- Currently defines schema version "0.1" with stages:
    "init", "device", "target", "fixed_eq", "free_eq",
    "optimization_fit", "complete"

Important: This is the canonical place to evolve the data contract.

Public API
----------
1) validate_h5_structure(h5_path: PathLike, schema_version: str, stage: str,
                         *, allow_extra: bool = True) -> None
   - Opens file read-only, checks all required paths exist.
   - Raises:
       FileNotFoundError: file missing
       ValueError: unknown schema_version or stage
       RuntimeError: missing required paths (with formatted message)
   - Also *optionally* checks /meta/schema_version matches requested version
     if present, but intentionally keeps this robust (won’t crash on weird storage).

2) require_groups(h5: h5py.File, groups: Iterable[str]) -> None
   - For checking that specific paths exist AND are groups (not datasets).

3) list_missing_paths(h5: h5py.File, required_paths: Iterable[str]) -> List[str]
   - Utility returning missing paths.

Internal helpers (design-relevant)
----------------------------------
- `_hint_for_stage(stage)` maps stage to next script name:
    init  -> scripts/00_init_run.py
    device-> scripts/01_build_device.py
    target-> scripts/02_target_boundary.py
    fixed_eq-> scripts/03_solve_fixed_gs.py
    free_eq-> scripts/05_solve_free_gs.py (and implies 04_fit_pf_currents)
    report-> scripts/07_make_report.py
  Note: “report” is mentioned in hints but not currently a schema stage; current
  stage list includes “complete” instead.

Required paths (schema v0.1)
----------------------------
The file encodes the “expected HDF5 contract” for each stage. Highlights:

- init stage requires:
    /meta, /meta/schema_version, /meta/created_utc, /meta/run_id,
    /meta/git_commit, /meta/notes, /history

- device stage requires:
    /grid/{R,Z,RR,ZZ}
    /device/vessel_boundary
    /device/coils/{names,centers,radii,I_max,I_pf}
    /device/coil_greens/psi_per_amp

- target stage requires:
    /target/boundary, /target/psi_boundary
    /target/global_targets/enforce
    /target/global_targets/Ip_target

…and so on for fixed_eq, free_eq, optimization_fit, complete.

Dataflow role
-------------
This module is the *bridge* between “scripts as stages” and “results.h5 as API”.
It enables resumable workflows:
  • If a user points a later script at a run_dir, the script can fail fast with
    actionable messages if upstream stages weren’t run.

Dependencies
------------
- h5py
- stdlib: pathlib, typing


================================================================================
units.py — Central unit conventions (minimal)
================================================================================

Purpose
-------
Single place for unit conventions and future unit handling.

Current state
-------------
- Minimal implementation:
  • defines numerical conversion factors (m, cm, kA, MA, T, Pa, etc.)
  • assumes SI everywhere
  • provides trivial conversion helpers:
      to_SI(value, unit_factor)
      from_SI(value, unit_factor)
- Includes mu0 constant (duplicative with constants.py by comment) for semantic clarity.

Philosophy
----------
- “ALL unit logic goes here.”
- No random conversion constants sprinkled throughout scripts.
- Designed as an extension point for later integration with pint or similar.

Public API
----------
- constants: m, cm, mm, km, s, ms, us, A, kA, MA, T, mT, Pa, kPa, MPa, J, kJ, MJ, mu0
- functions:
    to_SI(value, unit_factor)
    from_SI(value, unit_factor)
    enable_pint()  # placeholder raising NotImplementedError

Dependencies
------------
Stdlib only (no imports)

Design notes
------------
- Today this is primarily documentation + a gentle discipline tool.
- When integrating pint later, this module should become the only location that
  introduces pint quantities, while the rest of the code can remain stable.


================================================================================
Cross-module contract summary (io folder)
================================================================================

What other layers should rely on:
- Run folder contract (paths.py):
    run_dir/
      inputs/      (copied YAML configs)
      figures/     (plots & report outputs)
      results.h5   (canonical data store)
      run.log      (log output)
      summary.csv  (later reporting/aggregation)

- Logging contract (logging_utils.py):
    logger name "tokdesign"
    consistent formatting; console + file

- HDF5 contract helpers (h5.py):
    normalized paths
    safe overwrite semantics
    predictable string handling
    optional provenance snapshots under /history/<stage>/<event_id>

- Structural validation contract (schema.py):
    per-stage required paths in results.h5
    readable failure messages + “run this script next” hints

- Unit discipline (units.py):
    unit factors + conversion stubs (future pint integration)


================================================================================
Action items / quick fixes worth doing now
================================================================================
1) Fix the formatter bug in logging_utils.setup_logger() when handlers already exist.
2) Decide whether schema stages should include “report” or keep “complete” only,
   and make hints/stage names consistent.
3) Consider strengthening assert_is_run_dir() optionally (check results.h5 exists)
   if you want safer “resume from run folder” behavior.
