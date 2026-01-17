docs/scripts/02_target_boundary.md
=================================
Script: scripts/02_target_boundary.py

Date: 2026-01-16


Purpose
-------
`02_target_boundary.py` defines the *desired plasma boundary* (“target LCFS”) and
stores it in `results.h5`.

It is the workflow’s bridge between:
  • high-level design targets (R0, a, κ, δ, Ip_target, etc.) expressed in YAML
and
  • concrete geometric + scalar constraints stored in the canonical artifact.

This stage does **not** solve an equilibrium. It creates the target against which
later equilibrium solutions and optimization are evaluated.


Why this stage exists (workflow philosophy)
-------------------------------------------
This script turns “I want a plasma like this” into a stable, explicit *contract*:

- a closed boundary polyline `(/target/boundary)`
- a consistent boundary flux value `(/target/psi_boundary)`
- global scalar targets (Ip_target, and which global targets to enforce)

Downstream solvers and optimizers consume these artifacts and never need to
re-interpret the YAML.


Inputs (CLI)
------------
Required:
  --run-dir <path>
      Existing run directory (created by 00_init_run.py)

Optional:
  --target <path>
      Target-equilibrium YAML. If omitted, archived copy in run_dir/inputs/ is used.

  --overwrite
      Allow overwriting existing /target/* datasets in results.h5

  --log-level <LEVEL>
      Logging verbosity (default: INFO)

  --write-psi-boundary-alias
      Additionally write an alias dataset `/target/psi_b`
      (backwards-compat / convenience for older code or quick plotting).


Inputs (YAML content used)
--------------------------
From target_equilibrium.yaml (conceptual):
1) Target boundary definition
   - typically Miller parameters:
       R0, a, kappa, delta
     plus optional discretization settings (npts)

2) Flux convention at boundary
   - psi_boundary (often set to 0.0 as a gauge choice)

3) Global targets and enforcement flags
   - Ip_target (total plasma current target)
   - enforce flags (which scalar constraints to enforce later)

Key point:
----------
This stage defines “what to aim for”, not “how to hit it”.


Outputs (HDF5: results.h5)
--------------------------
This script is responsible for fully populating the **schema stage "target"**.

Written datasets / groups (v0.1 intent):

Target geometry:
  /target/boundary
    shape: (N,2) closed polyline [[R,Z], ...]

Flux boundary condition:
  /target/psi_boundary
    scalar float (often 0.0)

Global targets:
  /target/global_targets/enforce
    stored as a dict-like structure (implementation may be attrs or dataset)
  /target/global_targets/Ip_target
    scalar float [A]

Optional alias:
  /target/psi_b    (if --write-psi-boundary-alias)

After writing, the script validates:
  schema.validate_h5_structure(..., stage="target")


Core algorithm / logic flow
---------------------------
1) Resolve run_dir and results.h5
   - paths.assert_is_run_dir(run_dir)

2) Setup logging
   - append to run_dir/run.log

3) Validate prerequisites
   - schema.validate_h5_structure(..., stage="device")
     (target is downstream of device in the canonical workflow)

4) Load target config
   - Prefer archived YAML in run_dir/inputs/

5) Construct target boundary polyline
   - geometry.plasma_boundary.miller_boundary(...)
   - boundary is closed, validated, SI meters

6) Determine psi_boundary
   - from YAML (often 0.0)
   - written as scalar dataset

7) Write global targets
   - Ip_target (scalar)
   - enforce flags (which constraints are active)

8) Persist to HDF5 (/target/*)
   - use io.h5 writers
   - respect --overwrite

9) Optionally write alias dataset
   - /target/psi_b = /target/psi_boundary

10) Validate schema for stage="target"


Physics and numerical meaning
-----------------------------
This script encodes *geometric constraints* and *scalar targets* that later stages
use in two distinct ways:

1) Fixed-boundary GS solve (03_solve_fixed_gs.py)
   - uses /target/boundary as the prescribed LCFS region
   - uses /target/psi_boundary as the Dirichlet boundary value for ψ

2) Optimization / fitting (04_fit_pf_currents.py and beyond)
   - uses /target/boundary as a “shape objective / constraint”
   - uses /target/global_targets/* as scalar objectives

Numerical aspects here are simple:
- boundary is discretized at N points (default ~400)
- no PDE solving, no iteration
- accuracy depends on boundary discretization density only


Dependencies
------------
Internal:
  tokdesign.geometry.plasma_boundary
  tokdesign.io.paths
  tokdesign.io.h5
  tokdesign.io.schema
  tokdesign.io.logging_utils

Third-party:
  numpy
  h5py
  yaml

Stdlib:
  argparse, pathlib, typing


Error handling / failure modes
------------------------------
- Missing device stage:
    schema validation fails early.
- Invalid Miller parameters (negative a, kappa <= 0, etc.):
    geometry module raises ValueError.
- Existing /target datasets without --overwrite:
    HDF5 writes refuse to overwrite.
- Missing required config keys (Ip_target, enforce flags):
    KeyError or validation error (depending on implementation).


Relationship to later stages
----------------------------
Downstream scripts assume:
  • /target/boundary exists and is closed
  • /target/psi_boundary exists (scalar)
  • global targets exist (Ip_target at minimum)

In particular, 03_solve_fixed_gs.py requires:
  • /grid/* (from device stage)
  • /device/coil_greens/psi_per_amp (from device stage)
  • /device/coils/I_pf (from device stage)
  • /target/boundary and /target/psi_boundary (from this stage)


Practical usage patterns
------------------------
A) Typical run via run_workflow.py:
  run_workflow.py ... --to 02

B) Standalone (tuning boundary parameters):
  python scripts/02_target_boundary.py \
    --run-dir data/runs/<run_id> \
    --target configs/target_equilibrium.yaml \
    --overwrite

C) Backwards-compat alias writing:
  python scripts/02_target_boundary.py ... --write-psi-boundary-alias


What is intentionally NOT done here
-----------------------------------
- No equilibrium solve (no Grad–Shafranov)
- No coil current fitting
- No field computation
- No plotting (handled by quicklook/report scripts)

This script defines the *target*, not the *solution*.


Brief notes on future extensions
--------------------------------
Likely evolutions that still belong here:
  • alternative boundary parameterizations (e.g. explicit point sets, spline, X-point)
  • additional global targets (β_p, li, q95, shape moments)
  • storing more derived boundary metrics (area, kappa_est, delta_est) for diagnostics

Anything that requires solving for ψ must stay in physics + solver stages.
