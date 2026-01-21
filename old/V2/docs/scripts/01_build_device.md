docs/scripts/01_build_device.md
==============================
Script: scripts/01_build_device.py

Date: 2026-01-16


Purpose
-------
`01_build_device.py` constructs the *static machine description* of the tokamak
and stores it in `results.h5`.

This includes:
  • the computational R–Z grid
  • the vacuum vessel (wall) geometry
  • PF coil geometry and current limits
  • precomputed coil → ψ Green’s functions

This script defines everything that is *time-independent and plasma-independent*
about the machine. All later stages (targets, equilibria, optimization) rely on
these artifacts.


Why this stage exists (workflow philosophy)
-------------------------------------------
This stage freezes the **device geometry contract**:

- Geometry is created once and then treated as immutable.
- Downstream scripts must not recompute or reinterpret geometry.
- All physics and optimization operate *on top of* this frozen description.

This separation is crucial for:
  • reproducibility
  • debugging (geometry vs physics errors)
  • performance (Green’s functions are expensive; compute once)


Inputs (CLI)
------------
Required:
  --run-dir <path>
      Path to an existing run directory created by 00_init_run.py

Optional:
  --device <path>
      Path to baseline device YAML.
      If omitted, the script will use the archived copy in run_dir/inputs/.

  --overwrite
      Allow overwriting existing geometry datasets in results.h5.

  --log-level <LEVEL>
      Logging verbosity (default: INFO)

Notes:
------
- The script *always* prefers the archived config in run_dir/inputs/ if present.
- Passing --device explicitly is mainly for standalone use outside run_workflow.py.


Inputs (YAML content used)
--------------------------
From baseline_device.yaml (conceptual groups):

1) Grid definition:
   - R_min, R_max
   - Z_min, Z_max
   - NR, NZ

2) Vessel definition:
   - representation: "parametric" or "polyline"
   - parameters or explicit polyline points

3) Coil definition:
   - pf_coils.coils[*]: name, Rc, Zc, a, I_init, I_max
   - central_solenoid: enabled, Rc, Zc, a, I_init, I_max

No plasma physics parameters are used here.


Outputs (HDF5: results.h5)
--------------------------
This script is responsible for fully populating the **schema stage "device"**.

Written datasets / groups (v0.1 intent):

Grid:
  /grid/R
  /grid/Z
  /grid/RR
  /grid/ZZ

Vessel:
  /device/vessel_boundary

Coils:
  /device/coils/names
  /device/coils/centers        [[Rc,Zc], ...]
  /device/coils/a              (size proxy)
  /device/coils/I_pf           (initial currents)
  /device/coils/I_max

Coil Green’s functions:
  /device/coil_greens/psi_per_amp
    shape: (Ncoils, NZ, NR)

Attributes:
  - units attached where appropriate (meters, amperes)

After writing, the script validates:
  schema.validate_h5_structure(..., stage="device")


Core algorithm / logic flow
---------------------------
1) Resolve run_dir and results.h5
   - paths.assert_is_run_dir(run_dir)
   - open results.h5 in append mode

2) Setup logging
   - append logs to run_dir/run.log

3) Validate prerequisites
   - schema.validate_h5_structure(..., stage="init")

4) Load baseline device config
   - Prefer archived YAML in run_dir/inputs/

5) Build grid (geometry/grids.py)
   - make_rz_grid(...)
   - optional grid_spacing sanity check

6) Build vessel boundary (geometry/vessel.py)
   - load_vessel_from_config(...)
   - ensure closed, validated polyline

7) Build PF coil objects (geometry/coils.py)
   - coils_from_config(...)
   - returns ordered list of PFCoil objects

8) Precompute coil Green’s functions
   - compute_coil_psi_greens(coils, RR, ZZ)
   - expensive step (O(Ncoils × NR × NZ))

9) Write all geometry artifacts to HDF5
   - using io.h5_* helpers
   - overwrite controlled by --overwrite

10) Validate schema for stage="device"
    - ensures all required geometry exists


Numerical and algorithmic considerations
----------------------------------------
Grid:
  • uniform rectangular grid
  • R strictly positive (enforced by geometry)

Vessel boundary:
  • purely geometric
  • used later for masks and clearance constraints

Coil Green’s functions:
  • analytic evaluation via elliptic integrals
  • stored per ampere → enables linear recombination
  • dominates runtime of this script

Performance notes:
  • this stage is intentionally “heavy”
  • downstream stages rely on its outputs to be fast


Dependencies
------------
Internal:
  tokdesign.geometry.grids
  tokdesign.geometry.vessel
  tokdesign.geometry.coils
  tokdesign.io.paths
  tokdesign.io.h5
  tokdesign.io.schema
  tokdesign.io.logging_utils

Third-party:
  numpy
  h5py
  yaml
  scipy (indirectly, via elliptic integrals in greens)

Stdlib:
  argparse, pathlib, typing


Error handling / failure modes
------------------------------
- Missing init stage:
    schema validation fails early.
- Invalid geometry parameters:
    geometry modules raise ValueError with explicit messages.
- Existing geometry without --overwrite:
    h5 write helpers refuse to overwrite.
- Coil Green’s function computation failure:
    typically due to invalid grid or coil placement.


Relationship to later stages
----------------------------
Downstream scripts assume:
  • geometry datasets are present and immutable
  • coil ordering is fixed and consistent
  • coil Green’s functions exist

Specifically:
  - 02_target_boundary.py uses grid extents for sanity
  - 03_solve_fixed_gs.py uses:
        /grid/*
        /device/vessel_boundary
        /device/coil_greens/psi_per_amp
        /device/coils/I_pf


Practical usage patterns
------------------------
A) Typical run via run_workflow.py:
  run_workflow.py ... --to 01

B) Standalone (debugging geometry):
  python scripts/01_build_device.py \
    --run-dir data/runs/<run_id> \
    --device configs/baseline_device.yaml \
    --overwrite

C) Iterating on device geometry:
  - edit baseline_device.yaml
  - create a NEW run (recommended), or
  - reuse run_dir with --overwrite (advanced / careful)


What is intentionally NOT done here
-----------------------------------
- No plasma boundary definition
- No Grad–Shafranov solve
- No optimization
- No plotting (except via quicklook later)

This script defines *what the machine is*, not *what the plasma does*.


Brief notes on future extensions
--------------------------------
Possible future additions (still belong here):
  • non-uniform grids
  • shaped coils (beyond filament approximation)
  • multiple vessel components

Anything that *depends on plasma state* must stay out of this script.
