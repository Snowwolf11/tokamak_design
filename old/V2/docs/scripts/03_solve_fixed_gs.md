docs/scripts/03_solve_fixed_gs.md
================================
Script: scripts/03_solve_fixed_gs.py

Date: 2026-01-16


Purpose
-------
`03_solve_fixed_gs.py` computes a *fixed-boundary Grad–Shafranov equilibrium* on
the previously defined grid, using the previously defined target LCFS as the
plasma domain boundary.

It is the first stage that produces an equilibrium ψ field and (optionally)
derived diagnostics.

“Fixed-boundary” means:
  • the plasma boundary is prescribed (from /target/boundary)
  • ψ on that boundary is prescribed (from /target/psi_boundary)
  • the solver does not move the boundary to satisfy force balance with external
    coils (that is deferred to free-boundary GS later)


High-level idea (physics + numerics)
------------------------------------
Inside the target LCFS, solve:

  Δ* ψ_plasma = - μ0 R j_phi(ψ_total)

with:
  ψ_total = ψ_vac + ψ_plasma

Boundary condition (Dirichlet):
  ψ_total = ψ_boundary  on the LCFS

In the current implementation the solve is effectively linearized:
  • ψ_vac is computed from coil Green’s functions and coil currents
  • j_phi is evaluated from a profile model using available ψ (often ψ_vac),
    so the solver performs a single sparse linear solve rather than a nonlinear
    Picard/Newton loop

Numerical method summary:
  • uniform-grid finite differences for Δ*
  • sparse matrix assembly for interior nodes
  • Dirichlet boundary enforcement via RHS modification
  • direct sparse solve (SciPy spsolve)


Inputs (CLI)
------------
Required:
  --run-dir <path>
      Existing run directory

Optional:
  --target <path>
      Target-equilibrium YAML (otherwise use archived copy)

  --solver <path>
      Solver YAML (otherwise use archived copy)

  --overwrite
      Allow overwriting existing /equilibrium/* datasets

  --log-level <LEVEL>
      Logging verbosity (default: INFO)


Inputs (HDF5 prerequisites)
---------------------------
This script consumes artifacts produced by stages 01 and 02.

Required reads (conceptually):
From /grid (device stage):
  /grid/R
  /grid/Z
  /grid/RR
  /grid/ZZ

From coils (device stage):
  /device/coil_greens/psi_per_amp    shape (Ncoils, NZ, NR)
  /device/coils/I_pf                 shape (Ncoils,)

From target (target stage):
  /target/boundary                   shape (N,2)
  /target/psi_boundary               scalar

(Depending on implementation, it may also use vessel boundary for masks or checks:
  /device/vessel_boundary )


Inputs (YAML content used)
--------------------------
From solver.yaml (conceptual):
  • profile configuration (choice of j_phi model, parameters, normalization)
  • numerical options (tolerances, any optional iteration flags)
  • any explicit total plasma current target usage

From target_equilibrium.yaml (if not fully stored in HDF5):
  • may re-read psi_boundary / boundary settings, but the intended contract is
    “HDF5 is the API”, so scripts should prefer HDF5 values when present.


Outputs (HDF5: results.h5)
--------------------------
This script is responsible for producing the **schema stage "fixed_eq"**.

Minimum expected outputs (typical):
  /equilibrium/psi               (ψ_total on full grid, NZ×NR)

Often also useful to store explicitly:
  /equilibrium/psi_vac           (vacuum flux, NZ×NR)
  /equilibrium/psi_plasma        (plasma contribution, NZ×NR)

Optional diagnostic outputs (if implemented here or via derived/fields modules):
  /fields/B_R, /fields/B_Z
  /derived/magnetic_axis/R
  /derived/magnetic_axis/Z
  /derived/psi_axis
  /derived/psi_boundary

After writing, the script should validate:
  schema.validate_h5_structure(..., stage="fixed_eq")
(Exact required paths depend on schema v0.1 in your io/schema.py.)


Core algorithm / logic flow (script level)
------------------------------------------
1) Resolve run_dir and results.h5
   - paths.assert_is_run_dir(run_dir)
   - open results.h5 in append mode

2) Setup logging
   - append to run_dir/run.log

3) Validate prerequisites
   - schema.validate_h5_structure(..., stage="target")
     (fixed-boundary solve assumes device + target exist)

4) Load configs
   - prefer archived YAMLs from run_dir/inputs:
       target.yaml, solver.yaml

5) Load geometry + targets from HDF5
   - R, Z, RR, ZZ
   - target boundary polyline
   - psi_boundary scalar

6) Compute vacuum ψ from coils (linear superposition)
   - read G_psi = /device/coil_greens/psi_per_amp
   - read I_pf  = /device/coils/I_pf
   - compute:
       psi_vac = tensordot(I_pf, G_psi, axes=(0,0))

7) Call fixed-boundary GS solver (physics layer)
   - psi_total = physics.gs_solve_fixed.solve_fixed_boundary(
         R, Z, RR, ZZ,
         psi_vac,
         plasma_boundary,
         profile_cfg
     )

8) Optionally compute fields and derived quantities
   - fields.compute_poloidal_field(R, Z, psi_total)
   - derived.find_magnetic_axis(RR, ZZ, psi_total)
   - derived.psi_at_boundary(psi_total, boundary)

9) Write outputs to HDF5
   - /equilibrium/psi (and optionally psi_vac, psi_plasma)
   - /fields/*, /derived/* (if computed)
   - respect --overwrite

10) Validate schema for stage="fixed_eq"


Numerical algorithms used (detail)
----------------------------------

A) Masking the plasma domain (geometry-driven)
----------------------------------------------
To solve inside the LCFS, the solver needs a mask of interior points:
  inside(R_i, Z_j) = True if point is inside boundary polyline.

Typical algorithm:
  • point-in-polygon (ray casting) on flattened grid points
  • reshape mask back to (NZ, NR)

This domain mask determines which grid nodes appear in the sparse system.

B) Discretizing the Grad–Shafranov operator (finite differences)
---------------------------------------------------------------
Operator:
  Δ* ψ = R ∂/∂R(1/R ∂ψ/∂R) + ∂²ψ/∂Z²

Discretization:
  • second-order central differences on uniform grid
  • conservative form in R for stability/symmetry
  • sparse matrix with ~5 non-zeros per row for interior nodes

C) Applying Dirichlet boundary conditions
-----------------------------------------
Fixed-boundary means ψ_total is fixed on the LCFS.

Practically on a Cartesian grid with an arbitrary polyline boundary:
  • interior solve typically enforces ψ on boundary-adjacent nodes
  • boundary values enter via RHS correction:
      L ψ_interior = RHS_interior - L_boundary * ψ_boundary

The implementation details live in physics.gs_solve_fixed,
but at the script level the contract is:
  • pass boundary polyline + psi_boundary scalar
  • receive ψ_total on the full grid

D) Linear system solve
----------------------
Current solver strategy:
  • direct sparse solve: scipy.sparse.linalg.spsolve

Properties:
  • robust for medium-size problems
  • cost grows superlinearly; may become expensive for large grids
  • good baseline for correctness and debugging

Future swap (without changing script API):
  • iterative Krylov solve (CG/GMRES) + preconditioning
  • operator caching across iterations

E) Profile evaluation (source term)
-----------------------------------
The RHS depends on j_phi which is modeled in gs_profiles.

Current implementation behaves like:
  • evaluate j_phi from a prescribed analytic profile in normalized ψ
  • scale to meet Ip_target (if configured)
  • treat the resulting j_phi field as known for the single linear solve

This avoids nonlinear Picard/Newton loops for now.


Dependencies
------------
Internal:
  tokdesign.io.paths
  tokdesign.io.h5
  tokdesign.io.schema
  tokdesign.io.logging_utils
  tokdesign.physics.gs_solve_fixed
  tokdesign.physics.fields         (optional)
  tokdesign.physics.derived        (optional)

Third-party:
  numpy
  h5py
  yaml
  scipy.sparse / scipy.sparse.linalg (inside physics modules)

Stdlib:
  argparse, pathlib, typing


Error handling / failure modes
------------------------------
- Missing prerequisite stages:
    schema validation fails early with actionable message.
- Coil arrays inconsistent (I_pf length != Ncoils in G_psi):
    tensordot fails or explicit check fails.
- Boundary polyline invalid (not closed, R<=0, self-intersections):
    geometry validation may raise or solver may fail to build mask.
- Grid non-uniform:
    gs_operator may raise or assert (depending on implementation).
- Existing /equilibrium outputs without --overwrite:
    HDF5 writes refuse to overwrite.
- Solver numerical failure:
    sparse solve may raise MatrixRankWarning/RuntimeError (ill-posed domain).


Relationship to later stages
----------------------------
Outputs from this stage feed:
  • quicklook plotting (09_quicklook.py)
  • PF current fitting / optimization (04_fit_pf_currents.py)
  • future free-boundary iteration as an initial guess (05_solve_free_gs.py)


Practical usage patterns
------------------------
A) Typical run via run_workflow:
  run_workflow.py ... --to 03

B) Standalone solve (iterating on solver.yaml):
  python scripts/03_solve_fixed_gs.py \
    --run-dir data/runs/<run_id> \
    --solver configs/solver.yaml \
    --overwrite

C) Debugging:
  - confirm vacuum ψ looks correct by writing /equilibrium/psi_vac
  - plot ψ contours before attempting derived quantities


Brief notes on what is “yet to be implemented”
----------------------------------------------
- True nonlinear fixed-boundary solve:
    j_phi(ψ_total) → requires Picard/Newton iteration.
- Free-boundary GS:
    boundary is unknown and must be solved for with plasma-vacuum coupling.
- X-point / separatrix handling:
    target LCFS may include X-point geometry; requires more advanced boundary logic.
- More advanced profile physics (pressure + FF' consistency, bootstrap current, etc.)

The current stage is intentionally a robust baseline that keeps the script API stable.
