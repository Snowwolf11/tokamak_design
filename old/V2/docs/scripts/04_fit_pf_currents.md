docs/scripts/04_fit_pf_currents.md
=================================
Script: scripts/04_fit_pf_currents.py

Date: 2026-01-16


Purpose
-------
`04_fit_pf_currents.py` fits PF coil currents so that the *vacuum* poloidal flux
from the coils matches a target flux condition on the target LCFS.

This is a classic tokamak “shape control / boundary fitting” step:
  • actuator: PF coil currents (vacuum field only, no plasma current)
  • constraint/objective: flux behavior on boundary points of the target LCFS

It produces:
  1) updated “canonical” PF currents at:
       /device/coils/I_pf
  2) a detailed fit record under:
       /optimization/fit_results/*

and it snapshots previous values under /history for provenance.


High-level physics
------------------
Vacuum poloidal flux is linear in coil currents:

  ψ_vac(R,Z) = Σ_c G_ψ[c](R,Z) · I[c]

where G_ψ[c] is the coil Green’s function “ψ per ampere” for coil c, precomputed
during 01_build_device and stored as:

  /device/coil_greens/psi_per_amp   shape (Nc, NZ, NR)

The target file defines a desired LCFS boundary polyline and a boundary flux
value (often a gauge choice such as 0.0):

  /target/boundary       boundary points (Nb,2)
  /target/psi_boundary   ψ_target (scalar)

This script chooses I such that ψ_vac evaluated on the boundary points is
“as close as possible” to the target condition, with regularization and (optional)
bounds based on coil limits.


Inputs (CLI)
------------
Required:
  --run-dir <path>
      Existing run directory containing results.h5

  --solver <path>
      Path to solver.yaml (note: unlike earlier scripts, this one requires an
      explicit path; run_workflow.py typically passes the archived one)

Optional:
  --overwrite
      Allow overwriting existing /optimization/fit_results and /device/coils/I_pf

  --log-level <LEVEL>
      Logging verbosity (default: INFO)


Inputs (HDF5 prerequisites)
---------------------------
Requires that stages "device" and "target" exist:

From /grid (device stage):
  /grid/R, /grid/Z

From coils (device stage):
  /device/coil_greens/psi_per_amp   (Nc, NZ, NR)
  /device/coils/I_max               (Nc,)

From target (target stage):
  /target/boundary                  (Nb,2)
  /target/psi_boundary              scalar

Important:
- It does not require a GS equilibrium (/equilibrium/psi) because it fits vacuum
  flux only.


Inputs (solver.yaml: coil_fit section)
--------------------------------------
Reads `solver.yaml` and extracts:

coil_fit:
  method: str
      "ridge" (default) or "contour" or "contour_qp"
      (script warns if unknown but continues)

  reg_lambda: float
      ridge regularization strength (>=0)

  boundary_fit_points: int | None
      downsample number of boundary points (>=16 recommended)
      if None, uses all boundary points (minus closure duplicate)

  weight_by_coil_limits: bool
      if True, fit uses scaled variables x = I/I_max (recommended)

  enforce_bounds: bool
      if True, enforce current bounds via I_max (or x ∈ [-1,1])

  bounds_method: str
      "clip" (default) plus placeholders like "clip_then_refit", "active_set", "box"
      (warnings only; actual enforcement is inside optimization.coil_fit)

  psi_ref: float
      used for contour-type methods

  constraint: str
      used for contour-type methods ("mean" default; passed through)

Notes:
- Some options are “method-dependent” and may be ignored depending on the
  underlying optimization implementation in tokdesign.optimization.coil_fit.


Outputs (HDF5)
--------------
This script updates and writes:

A) Canonical updated PF currents
--------------------------------
/device/coils/I_pf               (Nc,)   [A]
  - overwritten unconditionally once the fit succeeds
  - a /history snapshot is written before overwriting

B) Fit results record
---------------------
/optimization/fit_results/
  I_pf_fit            (Nc,)      [A]
  x_fit               (Nc,)      [I/I_max]  only if weight_by_coil_limits=True
  boundary_points     (Nb_fit,2) [m]
  psi_boundary_fit    (Nb_fit,)  [Wb/rad]
  residual            (Nb_fit,)  [Wb/rad]
  residual_rms        scalar     [Wb/rad]   meaning depends on method (see below)

  A                   (Nb_fit, Nc)
      A[k,c] = ψ_per_amp at boundary point k from coil c
      (method-dependent scaling; see “A matrix construction”)

  b                   (Nb_fit,)
      reference vector used by fit (method-dependent)

  clamped             (Nc,) uint8
      1 where coil ended at a bound (as reported by optimizer)

  offset              scalar     [Wb/rad]   (boundary-value mode)
  psi_boundary_std    scalar     [Wb/rad]
  psi_boundary_ptp    scalar     [Wb/rad]

  contour_rms         scalar     [Wb/rad]   (only for contour methods)
  psi_ref             scalar     [Wb/rad]   (only for contour methods)
  attrs:
      method, reg_lambda, weight_by_coil_limits, enforce_bounds, bounds_method,
      psi_target, Nb_fit, Nc, (plus constraint attr for contour)

C) History snapshots (provenance)
---------------------------------
Before writing new I_pf and fit results, it snapshots:
  /device/coils/I_pf
  /optimization/fit_results
into:
  /history/04_fit_pf_currents/<event_id>/...


Core algorithm / logic flow (script level)
------------------------------------------
1) Resolve run_dir, results.h5 and setup logging
   - asserts run_dir structure
   - requires results.h5 exists
   - logger writes to run_dir/run.log

2) Load solver.yaml and parse coil_fit configuration
   - validate reg_lambda >= 0
   - validate boundary_fit_points >= 16 (if provided)

3) Validate prerequisites (structural)
   - validate_h5_structure(..., stage="device")
   - validate_h5_structure(..., stage="target")

4) Read data from HDF5
   - R, Z
   - G_psi (Nc, NZ, NR)
   - I_max (Nc,)
   - target boundary polyline
   - psi_target scalar

5) Boundary point selection / downsampling
   - If boundary is closed (first==last), remove the duplicate closure point
   - If boundary_fit_points is set:
       `_resample_boundary()` selects ~uniform indices along polyline:
         idx = round(linspace(0, Nb-1, npts, endpoint=False))
         unique(idx)
     (This is index-based resampling, not arclength interpolation.)

6) Variable scaling and bounds (important)
   - If `weight_by_coil_limits=True`:
       Define scaled variables x = I / I_max.
       Then:
         ψ = Σ_c (G_c · I_max[c]) x_c
       Internally the script forms:
         G_use = G_psi * I_max[:,None,None]
       Bounds become:
         x ∈ [-1,1]  if enforce_bounds

   - Else (unscaled):
       G_use = G_psi
       Bounds become:
         I ∈ [-I_max, +I_max]  if enforce_bounds

   Motivation for scaling:
   - makes the regularization term and bounds “comparable” across coils of
     different strength/limit
   - reduces numerical ill-conditioning when some coils have huge I_max and others small

7) Fit currents via optimization module
   - Calls:
       tokdesign.optimization.coil_fit.fit_pf_currents_to_boundary(
           G_psi=G_use,
           boundary_pts=boundary_fit,
           R=R,
           Z=Z,
           psi_target=psi_target,
           reg_lambda=reg_lambda,
           I_bounds=bounds,
           method=method,
           fit_offset=True,
           psi_ref=psi_ref,
           constraint=constraint,
       )

   This function returns a dict that includes:
     A, b, I_fit, psi_boundary_fit, residual, residual_rms, clamped, etc.

8) Unscale back to physical currents (if using x)
   - The optimization returns “I_fit” in the variable it solved for.
   - If weight_by_coil_limits=True, interpret it as x_fit and compute:
       I_phys = x_fit * I_max
   - The script stores both:
       /optimization/fit_results/x_fit   (scaled)
       /optimization/fit_results/I_pf_fit and /device/coils/I_pf (physical)

9) Report diagnostic statistics
   - Logs std/ptp of ψ along boundary
   - Logs clamp count and current range
   - NOTE: the script currently prints a block of “DEBUG ...” statistics to stdout
     (A stats, b stats, solution norm, boundary range). This is helpful while
     developing but is noisy for production runs.

10) Snapshot + write outputs to HDF5
    - Creates /optimization/fit_results groups
    - Snapshots prior values under /history/04_fit_pf_currents/<event_id>/
    - Writes updated /device/coils/I_pf
    - Writes detailed fit results
    - Adds metadata as attrs


Numerical algorithms used (the important details)
-------------------------------------------------

A) Constructing the linear system at the boundary
------------------------------------------------
The fit is based on sampling ψ_vac at boundary points (Rk,Zk).

Given a boundary point k, the contribution from coil c is:
  ψ_kc = G_ψ[c](Rk,Zk)   [ψ per amp]

So the discretized model is:
  ψ_boundary ≈ A I

where:
  A[k,c] = G_c evaluated at boundary point k
  I[c]   = coil current

Key numerical step:
- evaluating G_ψ[c](Rk,Zk) requires interpolation from the stored grid (R,Z).
- that interpolation is performed inside `fit_pf_currents_to_boundary` (not here).
  (Typical choice is bilinear interpolation on the structured grid.)

B) Ridge-regularized least squares (method="ridge")
---------------------------------------------------
The classical objective is:

  min_I  ||A I - b||² + λ ||I||²

where typically:
  b = ψ_target * 1  (constant vector)

This script also supports an offset (fit_offset=True):
  min_{I,c} ||A I + c·1 - b||² + λ ||I||²

This is important because ψ has an arbitrary gauge; allowing an offset can make
the fit focus on *shape* rather than absolute flux level (depending on how b is defined).

Solving ridge LS is typically done by one of:
  • normal equations: (AᵀA + λI) I = Aᵀb
  • or an augmented system / QR/SVD
Which one is used depends on optimization.coil_fit (not in this script).

C) Contour-based fitting (method="contour" / "contour_qp")
----------------------------------------------------------
These methods aim to make the boundary an *iso-flux contour* rather than match an
absolute ψ_target:

  ψ_boundary(Rk,Zk) should be constant along k

A typical formulation is:
  minimize variance of ψ on boundary:
    min_I  ||(ψ_k - mean(ψ))||² + λ||I||²

or constrain mean(ψ) to a reference (psi_ref) and minimize deviations.

The script passes:
  psi_ref and constraint
and logs “contour_rms” when in contour mode.

D) Bound constraints (current limits)
-------------------------------------
Bounds reflect engineering limits:
  |I[c]| ≤ I_max[c]

The script supports:
- scaled bounds x ∈ [-1,1] when using weight_by_coil_limits
- physical bounds I ∈ [-I_max, I_max] otherwise

The actual bounded-solve algorithm is delegated to optimization.coil_fit.
The script warns if bounds_method is unfamiliar, but does not enforce it itself.

Common practical algorithms for bounded ridge LS include:
  • simple clipping ("clip") after an unconstrained solve
  • active-set / projected gradient methods
  • box-constrained quadratic programming (QP)
The method names suggest those are planned/partially implemented in coil_fit.


Dependencies
------------
Stdlib:
  argparse, pathlib, typing

Third-party:
  numpy
  yaml
  h5py

Internal:
  tokdesign.io.logging_utils.setup_logger
  tokdesign.io.paths.assert_is_run_dir
  tokdesign.io.h5: open_h5, h5_read_*, h5_write_*, h5_snapshot_paths, ...
  tokdesign.io.schema.validate_h5_structure
  tokdesign.optimization.coil_fit.fit_pf_currents_to_boundary   (core solver)


Gotchas / current rough edges
-----------------------------
1) Debug prints to stdout
   - The script prints several “DEBUG ...” blocks unconditionally.
   - This is great during development but will clutter logs in routine runs.
   Recommended: gate behind log-level DEBUG or a --debug flag.

2) Boundary resampling is index-based, not arclength-based
   - `_resample_boundary()` picks indices uniformly, which can overweight regions
     with denser original sampling.
   - For Miller boundary this is usually fine; for arbitrary polylines it may bias.
   Future improvement: resample by arclength.

3) /optimization/fit_results overwrite behavior
   - Without --overwrite, it refuses to run if fit_results already exists.
   - It still snapshots prior values before writing new results (good).

4) Interpretation of residual_rms depends on method
   - In contour mode it represents RMS of deviations in ψ along the boundary
     (or a contour_rms value).
   - In boundary-value mode it represents RMS of (ψ - target) including offset.
   The script stores both residual_rms and contour_rms (for contour).


Relationship to other stages
----------------------------
Prerequisites:
  • 01_build_device.py must have created coil greens + limits
  • 02_target_boundary.py must have created target boundary + psi_boundary

Downstream:
  • Updated /device/coils/I_pf will affect any subsequent vacuum ψ recomputation.
  • This is intended to provide a good starting point for a future free-boundary
    equilibrium stage (05_solve_free_gs.py), which is not implemented yet.


Brief notes on what is “yet to be implemented”
----------------------------------------------
- A fully robust constrained solver backend in tokdesign.optimization.coil_fit:
    * proper QP / active-set with guaranteed optimality for box constraints
    * consistent handling of offset + contour objectives
- Coupling to free-boundary equilibrium iteration:
    * use these fitted currents as initial guess
    * iterate between boundary, plasma current, and coil currents
- Better resampling/weighting strategies:
    * arclength resampling of boundary points
    * curvature-based weighting
    * weighting by diagnostic importance (e.g., X-point region)
