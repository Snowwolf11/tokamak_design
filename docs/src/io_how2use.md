docs/src/io_how2use.md
====================

16.01.26

Pass B: How the pipeline scripts should use the `src/.../io` layer

Context
-------
Pass A documented what the io modules *are* (API, responsibilities).
This Pass B documents how they’re meant to be *used from the scripts layer*.

The guiding workflow contract
-----------------------------
Across this repo, scripts are “stages” that:
  1) read configs (YAML) + a run directory (or results.h5)
  2) compute something (by calling `src/...` logic)
  3) write their outputs to the canonical artifact: `results.h5`
  4) optionally write plots to `run_dir/figures/`
  5) validate that the expected HDF5 paths now exist

The io layer exists so scripts don’t reimplement:
  • run folder creation and path handling (paths.py)
  • consistent logging (logging_utils.py)
  • HDF5 write/read boilerplate (h5.py)
  • stage prerequisites / outputs checks (schema.py)
  • unit conversion constants (units.py)


1) Standard run lifecycle (how scripts fit together)
----------------------------------------------------

A. Creating a new run (00_init_run.py)
--------------------------------------
Responsibilities:
  • create run_id + directory tree
  • copy input YAMLs into run_dir/inputs for provenance
  • initialize results.h5 with /meta and /history
  • store run metadata early (schema_version, created_utc, run_id, git_commit, notes)
  • set up logging to run_dir/run.log

Recommended sequence (conceptual):
  1) run_id = paths.make_run_id(prefix=...)
  2) p = paths.create_run_dir(run_root, run_id)
     - p["run_dir"], p["results_path"], p["log_path"], ...
  3) logger = logging_utils.setup_logger(p["log_path"], level=...)
  4) paths.copy_inputs([device.yaml, target.yaml, solver.yaml, ...], p["inputs_dir"])
  5) with h5.open_h5(p["results_path"], "a") as f:
        h5.h5_ensure_group(f, "/meta")
        h5.h5_ensure_group(f, "/history")
        h5.h5_write_scalar(f, "/meta/schema_version", "0.1")
        h5.h5_write_scalar(f, "/meta/created_utc", "<UTC ISO8601>Z")
        h5.h5_write_scalar(f, "/meta/run_id", run_id)
        h5.h5_write_scalar(f, "/meta/git_commit", "<hash or ''>")
        h5.h5_write_scalar(f, "/meta/notes", "<notes or ''>")
  6) schema.validate_h5_structure(p["results_path"], "0.1", "init")

Notes:
  • Even if git_commit/notes aren’t known, reserve them as empty strings;
    schema currently expects those paths to exist.
  • The io layer intentionally does not parse YAML. Provenance is a filesystem copy,
    while config validation belongs elsewhere.

B. Running later stages (01_build_device.py, 02_target_boundary.py, ...)
------------------------------------------------------------------------
Each later script should:
  1) accept a run_dir or results.h5 path via CLI
  2) verify run_dir looks valid (paths.assert_is_run_dir)
  3) set up logging to run_dir/run.log (append mode)
  4) validate prerequisites using schema.validate_h5_structure(...)
  5) do the stage computation
  6) write outputs via h5.* writers
  7) validate stage outputs using schema.validate_h5_structure(...)

The idea:
  • scripts communicate ONLY through results.h5
  • schema checks turn “implicit assumptions” into explicit errors with hints


2) Script-level patterns (recommended boilerplate)
--------------------------------------------------

Pattern A: “run_dir is the primary handle”
------------------------------------------
Most robust pattern in a pipeline repo:

Inputs:
  --run-dir <path>

Then derive:
  results_path = run_dir / "results.h5"
  log_path     = run_dir / "run.log"
  figures_dir  = run_dir / "figures"

Use:
  paths.assert_is_run_dir(run_dir)
  logger = setup_logger(log_path)
  validate_h5_structure(results_path, ...)

Why this pattern is preferred:
  • run_dir carries the entire execution context (inputs snapshot, figures, logs)
  • results.h5 alone is not enough for “nice” UX (plots/logs)

Pattern B: “results.h5 is primary (advanced)”
---------------------------------------------
Accept:
  --results <path/to/results.h5>

Then derive run_dir = results_path.parent.

This is fine, but:
  • you must still enforce that run_dir has inputs/figures (paths.assert_is_run_dir)
  • you lose the opportunity to clearly display “this run folder” as the unit of work

General best practice:
  • Internally, treat run_dir as primary.
  • Externally, allow both, but normalize to run_dir early.


3) Dataflow contract: how io enforces the “HDF5 as API” idea
------------------------------------------------------------

A. Writing data (h5.py)
-----------------------
When a stage produces arrays, store them as datasets:
  h5_write_array(f, "/grid/RR", RR, attrs={"units": "m"})
When it produces scalars:
  h5_write_scalar(f, "/meta/schema_version", "0.1")

Key behaviors scripts should rely on:
  • all paths normalized to leading “/”
  • parent groups auto-created
  • overwrite defaults to True for datasets/scalars (stage re-runs are supported)
  • strings stored as UTF-8 variable-length

Practical guidance:
  • Put “units” into attrs whenever you write physical quantities.
    (It’s lightweight and saves time later.)
  • Avoid dtype=object arrays; explicitly store strings via h5_write_strings.
  • If you need to store small metadata dicts, use h5_write_dict_as_attrs
    (but do not store large structures as attrs).

B. Snapshotting (h5_snapshot_paths)
-----------------------------------
Use snapshotting for provenance when a stage changes something important
(e.g., coil currents before/after fitting):

Example use cases:
  • before overwriting /device/coils/I_pf
  • before overwriting /equilibrium/psi_total

Conceptually:
  event_id = h5_snapshot_paths(
      f,
      stage="04_fit_pf_currents",
      src_paths=["/device/coils/I_pf", "/optimization"],
      attrs={"note": "before fit"}
  )

This gives you a built-in, file-local “version history” under:
  /history/<stage>/<event_id>/...


4) Dependency & validation contract (schema.py)
------------------------------------------------

A. Pre-flight checks (prerequisites)
------------------------------------
Before a stage runs, validate the upstream stage exists:

Examples (intended):
  • 01_build_device.py should validate "init"
  • 02_target_boundary.py should validate "device"
  • 03_solve_fixed_gs.py should validate "target"
  • 04_fit_pf_currents.py should validate "device" + "target" (depending on design)
  • 05_solve_free_gs.py should validate "optimization_fit" (or whatever you decide)

This ensures scripts fail fast with actionable messages.

B. Post-flight checks (outputs)
-------------------------------
After writing, validate the stage you just produced:

Example:
  schema.validate_h5_structure(results_path, "0.1", "device")

C. Important reality check (current code state)
------------------------------------------------
Current schema stages available in v0.1 are:
  init, device, target, fixed_eq, free_eq, optimization_fit, complete

But the hint table includes a "report" stage.
If scripts use "report" in calls to validate_h5_structure, it will raise
“Unknown stage”.

Action for scripts right now:
  • use "complete" as the final stage name (not "report")
  • or add "report" to the schema stages if that’s intended

(Keeping stage names consistent matters a lot, because this is the script/HDF5 contract.)


5) Logging contract (logging_utils.py)
--------------------------------------

Intended usage:
  logger = setup_logger(run_dir/"run.log", level="INFO")

Expectations:
  • log lines are timestamped and human-readable
  • logs go to stdout + run.log
  • scripts can be re-run and append to the same run.log

Known bug (must be avoided until fixed):
  setup_logger() currently breaks if called twice in the same Python process,
  because it references `formatter` before it’s defined in the “handlers exist” branch.

Practical workaround for scripts (until fixed):
  • ensure setup_logger is called exactly once per process (typical CLI runs are fine)
  • avoid running multiple stages in-process unless you patch logging_utils.py

If you have a driver script that calls multiple stages in one process
(run_workflow.py style), you should fix logging_utils first.


6) Units discipline (units.py)
------------------------------
How scripts should use units.py today:
  • primarily as a “conversion vocabulary” and to prevent random constants
  • optionally as a source for attrs["units"] strings you write into HDF5

Examples:
  R_m = to_SI(R_cm, cm)
  h5_write_array(..., attrs={"units": "m"})

Important:
  units.py is intentionally minimal; do not start adding ad-hoc conversions elsewhere.
  If a conversion is needed, add it here so the whole codebase stays consistent.


7) Concrete recommendations for script authors
----------------------------------------------

A. Minimal “script skeleton” responsibilities
---------------------------------------------
Every stage script should do, in this order:
  1) parse args
  2) normalize run_dir + results_path
  3) assert run_dir structure
  4) setup logger
  5) validate prerequisites via schema.py
  6) run computation via src/ logic
  7) write outputs via h5.py
  8) validate stage outputs via schema.py

B. What scripts must NOT do
---------------------------
  • define physics formulas or numerical algorithms inline
  • write random files outside the run_dir structure
  • store intermediate state in ad-hoc pickle/npy files outside results.h5
  • silently proceed if prerequisites are missing

C. HDF5 path stability is king
------------------------------
Once a path like "/device/coils/I_pf" exists, treat it as a stable API.
If you need to change structure:
  • bump schema version (e.g., 0.2)
  • provide backwards compatibility or a migration step


8) Quick checklist: does a stage behave “correctly”?
----------------------------------------------------
- [ ] Creates no outputs outside run_dir
- [ ] Logs both to console and run.log
- [ ] Reads only from results.h5 + configs (or inputs snapshot)
- [ ] Writes only to results.h5 + figures/
- [ ] Validates prerequisites before running
- [ ] Validates outputs after running
- [ ] Stores units as dataset attrs when relevant
- [ ] Can be re-run without manual cleanup (overwrite semantics)


Appendix: mapping io modules to typical stage scripts
-----------------------------------------------------
paths.py
  • 00_init_run.py (create run folder + copy configs)
  • all other scripts (assert_is_run_dir; derive paths)

logging_utils.py
  • all scripts (setup once early)

h5.py
  • all scripts that write/read results.h5
  • optional history snapshots in “mutating” stages

schema.py
  • pre-flight + post-flight validation in every stage script

units.py
  • anywhere values come in non-SI or where conversions are needed


Notes / Action items emerging from Pass B
-----------------------------------------
1) Fix logging_utils.setup_logger() handler branch bug (formatter undefined).
2) Decide whether schema stage should be "report" or keep "complete"; align hints.
3) Consider adding an optional stronger run_dir validator (results.h5 existence),
   but keep paths.assert_is_run_dir lightweight by default.
