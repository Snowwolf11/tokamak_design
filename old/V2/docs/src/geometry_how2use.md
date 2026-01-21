docs/src/geometry_how2use.md
=========================
Pass B: How the pipeline scripts *use* the `src/.../geometry/` layer

Date: 2026-01-16

Scope
-----
This document explains the *practical usage patterns* of the geometry layer from
the scripts and higher-level modules:
  • when each geometry module is used
  • what data is passed in and out
  • how geometry objects are persisted to results.h5
  • how geometry feeds into physics, solvers, and optimization

This is not a restatement of APIs (Pass A); it is the *workflow contract*.


================================================================================
1) Geometry’s role in the overall workflow
================================================================================

Geometry is the *first physics-adjacent layer* after io.

High-level flow:
  configs (YAML)
      ↓
  geometry (pure shapes, grids, coil placement)
      ↓
  physics (fields, equilibria)
      ↓
  optimization / analysis
      ↓
  viz / reports

Key idea:
  • geometry defines “where things are”
  • physics defines “what fields those things produce”

Scripts should never:
  • recompute geometry ad hoc
  • infer geometry implicitly from physics arrays
Instead, geometry should be computed once and written explicitly to results.h5.


================================================================================
2) Typical geometry usage by stage scripts
================================================================================

A. 01_build_device.py (primary geometry stage)
----------------------------------------------
This is the *main consumer* of geometry modules.

Responsibilities:
  • read baseline device config (YAML)
  • build grid, vessel, coils
  • precompute coil Green’s functions
  • persist everything to results.h5

Typical sequence:
  1) Parse device config
  2) Build computational grid          → grids.py
  3) Build vessel boundary              → vessel.py
  4) Build PF coil objects              → coils.py
  5) Precompute coil ψ Green’s function → coils.py (+ physics.greens)
  6) Write outputs to results.h5
  7) Validate “device” schema stage

Geometry modules involved:
  • grids.py
  • vessel.py
  • coils.py

B. 02_target_boundary.py (target geometry)
------------------------------------------
This stage defines the *desired plasma shape* (LCFS).

Responsibilities:
  • read target config (YAML)
  • generate LCFS boundary
  • compute derived shape metrics (κ, δ, area)
  • write target geometry to results.h5

Typical sequence:
  1) Parse target config
  2) Generate boundary polyline          → plasma_boundary.py
  3) Compute area / κ / δ                → plasma_boundary.py
  4) Write to results.h5
  5) Validate “target” schema stage

Geometry modules involved:
  • plasma_boundary.py

C. Solver stages (03_solve_fixed_gs.py, 05_solve_free_gs.py)
------------------------------------------------------------
Solver stages *consume* geometry but do not create it.

They read:
  • grid (R, Z, RR, ZZ)
  • coil Green’s functions
  • vessel boundary (for masks / constraints)
  • target LCFS boundary (for boundary conditions or penalties)

They must:
  • trust geometry invariants (shapes, closure, R>0)
  • not modify geometry datasets

Geometry modules involved:
  • grids.py (spacing checks)
  • vessel.py (inside/outside masks, clearance checks)
  • plasma_boundary.py (metrics, if needed)

D. Optimization / fitting stages (04_fit_pf_currents.py)
--------------------------------------------------------
Optimization modifies *currents*, not geometry.

However, geometry is still used to:
  • interpret Green’s functions
  • compute clearance constraints to vessel
  • compute shape metrics from LCFS

Typical pattern:
  • geometry arrays are read-only
  • new coil currents are written as a new dataset or overwrite /device/coils/I_pf
  • optional snapshot of previous currents using io.h5_snapshot_paths

Geometry modules involved:
  • coils.py (psi assembly helper)
  • vessel.py (distance checks)
  • plasma_boundary.py (shape metrics)


================================================================================
3) Concrete usage patterns by module
================================================================================

grids.py — practical usage
--------------------------
Used once, early, in device-building stage.

Canonical usage:
  R, Z, RR, ZZ = make_rz_grid(
      R_min=cfg["grid"]["R_min"],
      R_max=cfg["grid"]["R_max"],
      Z_min=cfg["grid"]["Z_min"],
      Z_max=cfg["grid"]["Z_max"],
      NR=cfg["grid"]["NR"],
      NZ=cfg["grid"]["NZ"],
  )

  dR, dZ = grid_spacing(R, Z)   # optional sanity check

Persistence (script responsibility):
  /grid/R
  /grid/Z
  /grid/RR
  /grid/ZZ
  attrs:
    units = "m"

Key invariants scripts rely on later:
  • RR.shape == ZZ.shape == (NZ, NR)
  • R strictly increasing, R[0] > 0
  • uniform spacing (unless explicitly changed later)

plasma_boundary.py — practical usage
------------------------------------
Used for *target* specification and shape diagnostics.

Canonical usage:
  boundary = miller_boundary(
      R0=cfg["target"]["R0"],
      a=cfg["target"]["a"],
      kappa=cfg["target"]["kappa"],
      delta=cfg["target"]["delta"],
      npts=cfg.get("npts", 400),
  )

  area = boundary_area(boundary)
  kappa_est, delta_est = boundary_kappa_delta(boundary)

Persistence:
  /target/boundary           (polyline)
  /target/area               (scalar)
  /target/kappa_est          (scalar)
  /target/delta_est          (scalar)

Script-level expectations:
  • boundary is closed
  • R>0 everywhere
  • first and last points identical

Important:
  • target boundary is a *desired shape*, not necessarily an equilibrium LCFS
  • solver stages must not assume perfect agreement with actual equilibrium

coils.py — practical usage
--------------------------
Used primarily in device-building and optimization stages.

Canonical usage in device build:
  coils = coils_from_config(device_cfg)

  G_psi = compute_coil_psi_greens(coils, RR, ZZ)

  names   = [c.name for c in coils]
  centers = coil_centers(coils)
  I_pf    = coil_currents(coils)
  I_max   = coil_current_limits(coils)

Persistence:
  /device/coils/names
  /device/coils/centers
  /device/coils/a                (or “radii” if you standardize naming)
  /device/coils/I_pf
  /device/coils/I_max
  /device/coil_greens/psi_per_amp

Canonical usage in optimization:
  psi_vac = psi_from_coils(G_psi, I_pf)

  new_coils = set_coil_currents(coils, I_new)

Important contracts:
  • ordering of coils is fixed and consistent across arrays
  • G_psi[k,:,:] corresponds to coil k in the same order as names/I_pf
  • geometry does not change during optimization — only currents do

vessel.py — practical usage
---------------------------
Used in device build and solver/optimization stages.

Canonical usage in device build:
  vessel_poly = load_vessel_from_config(device_cfg)

Persistence:
  /device/vessel_boundary

Canonical usage in solver/optimization:
  inside = point_in_polygon(vessel_poly, np.c_[RR.ravel(), ZZ.ravel()])
  inside = inside.reshape(RR.shape)

  dmin = min_distance_to_polyline(vessel_poly, points)

Typical uses:
  • build computational masks
  • enforce minimum LCFS-to-wall clearance
  • reject candidate equilibria violating vessel constraints

Script-level expectations:
  • vessel boundary is closed
  • orientation (CW/CCW) does not matter
  • classification near boundary is approximate (acceptable for masks)


================================================================================
4) HDF5 contract: geometry datasets as stable API
================================================================================

Once written, geometry datasets are treated as *read-only API* for downstream stages.

Stable geometry paths (v0.1 schema intent):
  /grid/R
  /grid/Z
  /grid/RR
  /grid/ZZ

  /device/vessel_boundary
  /device/coils/*
  /device/coil_greens/psi_per_amp

  /target/boundary

Rules:
  • Scripts must not silently change geometry once “device” stage is validated.
  • If geometry structure changes, schema version must be bumped.
  • Derived quantities (metrics, masks) should live in new groups, not overwrite geometry.


================================================================================
5) Common mistakes this pass is meant to prevent
================================================================================

❌ Recomputing grids in solver scripts
   → Grid must be created once and persisted.

❌ Inferring coil ordering implicitly
   → Always rely on stored arrays (names, centers, I_pf).

❌ Mixing geometry and physics
   → No field solving or Grad–Shafranov math in geometry.

❌ Writing geometry to ad-hoc files
   → Geometry lives in results.h5 only.

❌ Treating LCFS target as an equilibrium
   → Target is a constraint, not a solution.


================================================================================
6) Quick checklist for script authors (geometry usage)
================================================================================

- [ ] Geometry is created only in designated stages (device, target)
- [ ] Geometry arrays have consistent shapes and units
- [ ] Geometry is written once and then treated as immutable
- [ ] Coil ordering is stable and documented
- [ ] Vessel boundary is closed and validated
- [ ] All geometry used downstream comes from results.h5, not recomputation


================================================================================
Notes / Action items emerging from Pass B
================================================================================
1) Standardize naming of coil size:
   - PFCoil.a vs HDF5 dataset name (“a” vs “radii”)
   - choose one and encode in schema to avoid ambiguity.

2) Consider adding a small “geometry reader” helper (optional):
   - e.g. load_grid(h5), load_coils(h5), load_vessel(h5)
   - would reduce script boilerplate and centralize conventions.

3) If you later add shaped grids or nonuniform spacing:
   - add new geometry functions; do not overload make_rz_grid silently.
