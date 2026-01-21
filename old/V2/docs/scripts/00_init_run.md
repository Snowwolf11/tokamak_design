docs/scripts/00_init_run.md
==========================
Script: scripts/00_init_run.py

Date: 2026-01-16


Purpose
-------
`00_init_run.py` initializes a fresh, self-contained “run” folder and a new
`results.h5` file.

It is the *first* stage in the workflow and defines the basic invariants that
all later stages rely on:

  • standard run directory layout
  • archived input configs for provenance
  • canonical HDF5 file exists and contains /meta/* and /history
  • schema validation for stage="init" passes


Why this exists (workflow philosophy)
-------------------------------------
This script makes every run reproducible and portable:

- **Reproducible inputs**: YAML configs are copied into `run_dir/inputs/`.
  Later stages (and `run_workflow.py`) can prefer the archived configs, so reruns
  are stable even if the original YAMLs change.

- **Provenance in one place**: key metadata is written into `results.h5:/meta`,
  making it easy to track how a result was generated.

- **Fail fast**: if anything about the run creation is broken, the workflow stops
  here rather than failing later in obscure ways.


Run directory structure (created here)
--------------------------------------
Creates:
  data/runs/<run_id>/
      inputs/
      figures/
      run.log
      results.h5
      summary.csv   (reserved by paths.create_run_dir; not created here)

Notes:
- `summary.csv` is not written by this script; the path is reserved for later
  reporting/aggregation.


Inputs (CLI)
------------
Required:
  --device <path>   baseline_device.yaml
  --target <path>   target_equilibrium.yaml
  --solver <path>   solver.yaml

Optional:
  --run-root <dir>          default: data/runs
  --prefix <str>            optional run id suffix/label
  --notes <str>             stored in /meta/notes
  --create-placeholders     pre-create empty groups in results.h5 for nicer browsing
  --log-level <LEVEL>       default: INFO

YAML reading in this script:
----------------------------
This stage loads the three YAML files *only* to:
  • infer schema_version
  • infer a default prefix from device_cfg.meta.name (if --prefix not given)

It does NOT validate or interpret physics/geometry settings beyond that.


Outputs (files, directories, HDF5 contents)
-------------------------------------------
Filesystem side effects:
1) Creates a new run directory tree under --run-root
2) Copies the three YAML config files into:
     run_dir/inputs/<basename>.yaml
3) Creates:
     run_dir/run.log (logging output)
     run_dir/results.h5

HDF5 contents written (guaranteed):
  /meta/schema_version   (string)
  /meta/created_utc      (string, ISO8601 UTC "YYYY-MM-DDTHH:MM:SSZ")
  /meta/run_id           (string)
  /meta/git_commit       (string, best effort)
  /meta/notes            (string)

  /history               (group)

Optional placeholder groups (if --create-placeholders):
  /grid
  /device
  /target
  /equilibrium
  /fields
  /derived
  /analysis
  /optimization

These placeholders are purely for convenience while browsing; they are not
required for stage="init" schema validation.


Core algorithm / logic flow
---------------------------
1) Resolve and load YAML configs
   - Uses yaml.safe_load into Python dicts.

2) Determine schema version
   - Priority order:
       device_cfg.meta.schema_version
       target_cfg.meta.schema_version
       solver_cfg.meta.schema_version
       default: "0.1"
   This is a “best effort” convention so configs can carry versioning without
   hardcoding it in scripts.

3) Determine run prefix
   - If --prefix provided: use that
   - Else: use device_cfg.meta.name (if present)
   - Else: no prefix

4) Create run_id and run_dir
   - run_id = tokdesign.io.paths.make_run_id(prefix)
     (timestamp + optional suffix)
   - run_dir structure created by tokdesign.io.paths.create_run_dir

5) Setup logger
   - tokdesign.io.logging_utils.setup_logger(run.log, level)
   - Writes consistent logs into run_dir/run.log and console.

6) Archive configs
   - tokdesign.io.paths.copy_inputs([...], run_dir/inputs)

7) Record provenance metadata in results.h5
   - created_utc: datetime.utcnow() formatted as ISO8601 Z
   - git_commit: retrieved via:
       git -C <project_root> rev-parse HEAD
     (returns "UNKNOWN" if not available)

8) Validate schema for stage="init"
   - tokdesign.io.schema.validate_h5_structure(results_path, schema_version, "init")
   - This ensures /meta/* and /history exist (and therefore later scripts can
     rely on them).

9) Print a handoff summary
   - Prints run_dir, results path, inputs dir, figures dir, log path, etc.


Dependencies
------------
Python stdlib:
  argparse, pathlib, datetime, subprocess, typing

Third-party:
  yaml (PyYAML)
  h5py

Internal:
  tokdesign.io.paths:
    - make_run_id
    - create_run_dir
    - copy_inputs
  tokdesign.io.logging_utils:
    - setup_logger
  tokdesign.io.h5:
    - h5_ensure_group
    - h5_write_scalar
  tokdesign.io.schema:
    - validate_h5_structure


Error handling / failure modes
------------------------------
- YAML missing / invalid:
    yaml.safe_load will raise; script exits.
- Run directory already exists:
    create_run_dir fails fast (intended).
- Not in a git repo / git unavailable:
    git_commit stored as "UNKNOWN" (non-fatal).
- HDF5 write failure:
    h5py.File(...,"w") raises; script exits.
- Schema mismatch:
    validate_h5_structure raises with explicit missing-path info.


Relationship to later stages
----------------------------
Downstream scripts assume:
  • run_dir exists and has inputs/ and figures/
  • results.h5 exists
  • results.h5 contains /meta and /history
  • schema_version is set

This script deliberately does NOT create:
  • /grid datasets
  • /device geometry
  • /target boundary
  • equilibrium solutions
Those are responsibilities of later pipeline stages.


Practical usage patterns
------------------------
A) Create a new run (typical):
  python scripts/00_init_run.py \
    --device configs/baseline_device.yaml \
    --target configs/target_equilibrium.yaml \
    --solver configs/solver.yaml \
    --prefix baseline \
    --notes "first try"

B) Create a run with placeholder groups (nice for browsing early):
  python scripts/00_init_run.py ... --create-placeholders

C) Using with run_workflow.py:
  run_workflow calls this stage first and then discovers the created run_dir.


Brief notes on what is “yet to be implemented”
----------------------------------------------
Nothing “physics heavy” is missing here; 00_init_run is largely complete.

Future enhancements (optional, workflow quality-of-life):
  • Emit the created run_dir in a machine-parseable way so run_workflow.py
    doesn’t have to guess “newest run dir”.
  • Record additional provenance:
      - Python package versions / pip freeze
      - hostname / OS
      - git diff status (“dirty” repo) or patch hash
  • Create summary.csv header row (currently just reserved).
