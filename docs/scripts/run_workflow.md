docs/scripts/run_workflow.md
============================
Script: scripts/run_workflow.py

Date: 2026-01-16

Purpose
-------
`run_workflow.py` is a *driver / orchestrator* that runs multiple pipeline stages
in sequence so the user doesn’t have to manually track the run directory or call
each stage script by hand.

It is intentionally “thin”:
  • contains NO physics, numerics, geometry building, or HDF5 logic
  • delegates all real work to stage scripts via subprocess calls
  • provides reproducibility conveniences (prefers archived configs in run_dir/inputs/)
  • optionally runs a quicklook report at the end

This makes it the preferred “one command” entry point during development.


High-level design philosophy (as implemented)
---------------------------------------------
1) Each stage script remains fully usable on its own
   - run_workflow only composes them; it does not replace them.

2) Stage-specific differences are isolated in small command builder functions
   - `_cmd_00`, `_cmd_01`, `_cmd_02`, `_cmd_03`, `_cmd_04`
   - The main loop is generic.

3) Reproducibility by default when re-running later stages
   - For each stage after 00, it uses `_archived_or_given(run_dir, given_path)`
   - That means: if `run_dir/inputs/<config>.yaml` exists, it uses that instead
     of the original file path passed on the command line.

4) Fail-fast execution
   - each stage is run with `subprocess.run(..., check=True)`
   - any non-zero exit code stops the workflow immediately.


Included stages and execution order
-----------------------------------
Stage order is enforced by `STAGE_ORDER`:

  STAGE_ORDER = ["00", "01", "02", "03", "04"]

Mapping to scripts (`STAGE_TO_SCRIPT`):
  "00" -> 00_init_run.py
  "01" -> 01_build_device.py
  "02" -> 02_target_boundary.py
  "03" -> 03_solve_fixed_gs.py
  "04" -> 04_fit_pf_currents.py
(future 05_solve_free_gs.py is mentioned but not registered)

Important: `--to` selects the final stage to run. The script then runs all
stages up to and including that stage in the fixed order above.


Inputs (CLI) and what they mean
-------------------------------
Required:
  --device <path>   baseline device YAML (passed into stage 00 and stage 01)
  --target <path>   target equilibrium YAML (passed into stage 00, 02, 03)
  --solver <path>   solver YAML (passed into stage 00, 03, 04)

Optional workflow controls:
  --run-root <dir>      where run directories live (default: data/runs)
  --prefix <str>        passed to 00_init_run.py (affects run_id naming)
  --notes <str>         passed to 00_init_run.py (stored in /meta/notes)
  --log-level <LEVEL>   passed to every stage script
  --to <00–04>          final stage to run (default "01")
  --overwrite           forwarded to stages 01–04 (not stage 00)
  --create-placeholders forwarded only to stage 00

Stage 02 passthrough:
  --write-psi-boundary-alias
    forwarded to 02_target_boundary.py

Quicklook controls:
  --no-quicklook
  --quicklook-formats png pdf svg ... (default: ["png"])
  --quicklook-dpi <int> (default: 160)
  --quicklook-greens-max <int> (default: 8)
  --quicklook-fieldlines
  --quicklook-fieldline-steps <int> (default: 2500)
  --quicklook-fieldline-ds <float>   (default: 0.01)

Note: quicklook is always run at the end unless `--no-quicklook` is set.


Outputs and side effects
------------------------
This script itself does not compute or write scientific outputs directly.
Its side effects are:

1) It launches stage scripts as subprocesses
   - their outputs are written into the run directory they operate on.

2) It determines the run_dir created by stage 00
   - via `_detect_newest_run_dir(run_root)`:
       chooses the most recently modified directory under run_root.

3) It prints status lines to stdout
   - e.g. the command being executed, and the detected run_dir.

4) It may run quicklook
   - which typically writes figures into run_dir/figures/ (handled by 09_quicklook.py)


Core internal functions (and why they exist)
--------------------------------------------

_run_subprocess(cmd: List[str]) -> None
  - prints command and executes it with check=True (fail-fast)

_detect_newest_run_dir(run_root: Path) -> Path
  - used immediately after stage 00 to find the new run directory
  - assumes “new run dir” is the most recently modified directory

_archived_or_given(run_dir: Path, given_path: str) -> Path
  - reproducibility helper
  - if run_dir/inputs/<basename> exists, return that
  - else return the resolved given_path

_scripts_dir() -> Path
  - returns directory containing run_workflow.py so stage script paths can be
    constructed reliably with relative filenames

_cmd_00/_cmd_01/_cmd_02/_cmd_03/_cmd_04(...)
  - stage-specific command builders that map workflow args into the exact CLI
    call needed for each stage
  - keeps the main loop generic

_run_quicklook(args, scripts_dir, run_dir)
  - runs 09_quicklook.py unless disabled
  - forwards plotting/format options and optional fieldline plotting options


Dataflow and dependencies (conceptual)
--------------------------------------
The script-level dataflow is purely “stage-to-stage via artifacts”:

  00_init_run.py
    creates run_dir/, results.h5, inputs snapshot, run.log, etc.

  01_build_device.py
    reads archived device.yaml
    writes geometry + coil greens to results.h5

  02_target_boundary.py
    reads archived target.yaml
    writes target LCFS / boundary datasets to results.h5

  03_solve_fixed_gs.py
    reads target.yaml + solver.yaml + results.h5 geometry
    writes equilibrium psi (and possibly fields/derived) to results.h5

  04_fit_pf_currents.py
    reads solver.yaml + results.h5 (greens, target constraints, etc.)
    writes updated currents / fit outputs to results.h5

Finally:
  09_quicklook.py
    reads results.h5
    writes plots into run_dir/figures/


Practical usage patterns
------------------------
A) Fresh run up to fixed-boundary equilibrium:
  python scripts/run_workflow.py \
    --device configs/baseline_device.yaml \
    --target configs/target_equilibrium.yaml \
    --solver configs/solver.yaml \
    --to 03

B) Iterate on a later stage (dev loop):
  - edit 03_solve_fixed_gs.py
  - re-run:
      python scripts/run_workflow.py ... --to 03 --overwrite

C) Preserve reproducibility:
  - If you rerun with changed config files on disk, run_workflow will still prefer
    the archived configs inside run_dir/inputs/ for stages 01–04.
  - That means re-running is stable even if the original YAMLs changed.


Failure modes and gotchas
-------------------------
1) “Newest directory detection” ambiguity
   - `_detect_newest_run_dir()` picks the most recently modified directory.
   - If another process writes into a different run folder at the same time, or if
     you manually touch run directories, it could pick the wrong one.
   Mitigation ideas:
     - Stage 00 could print the run_dir explicitly and this driver could parse it.
     - Or stage 00 could write a known marker file that run_workflow reads.

2) Stage availability vs `--to`
   - `--to` is validated against STAGE_TO_SCRIPT keys.
   - If you extend STAGE_ORDER but forget to extend STAGE_TO_SCRIPT/CMD_BUILDERS,
     you’ll get errors.

3) Stage 04 script existence
   - run_workflow includes stage 04, but if 04_fit_pf_currents.py isn’t present
     or not implemented in your repo snapshot, this workflow will fail at stage 04.

4) Logging setup “multiple calls in one process”
   - run_workflow calls stages as separate processes, so the known logger “double
     setup” issue in logging_utils.py is mostly avoided here.


What to implement next (brief)
------------------------------
When you add free-boundary GS later:
  • add "05" to STAGE_ORDER and STAGE_TO_SCRIPT
  • write a `_cmd_05(...)` command builder
  • add it to CMD_BUILDERS
The main loop won’t need changes.

If you add more “post stages” (reporting, export):
  • follow the same pattern (command builder + stage registry)
  • keep this driver free of physics logic.


One-line summary
----------------
`run_workflow.py` is the thin, reproducible stage orchestrator:
it runs 00→…→N in order, discovers the run_dir, prefers archived configs, and
(optionally) finishes with quicklook.
