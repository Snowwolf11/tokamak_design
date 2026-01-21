docs/src/physics_how2use.md
========================
Pass B: How the pipeline scripts *use* the `src/.../physics/` layer

Date: 2026-01-16

Scope
-----
This document explains how the physics modules are *used in practice* by the
pipeline scripts and higher-level workflow logic.

It complements Pass A by focusing on:
  • when and where physics code is called
  • which inputs come from geometry / HDF5
  • which outputs are written back to results.h5
  • how numerical solvers are orchestrated at the script level
  • what assumptions scripts are allowed to make

This document deliberately avoids re-deriving equations; those live in Pass A.


================================================================================
1) Physics’ role in the staged workflow
================================================================================

Physics sits between *geometry* and *analysis/optimization*.

Canonical stage flow:
---------------------
00_init_run
    ↓
01_build_device        (geometry + vacuum Green’s functions)
    ↓
02_target_boundary     (target LCFS geometry)
    ↓
03_solve_fixed_gs      (FIRST physics solve)
    ↓
04_fit_pf_currents     (optimization, uses physics linearly)
    ↓
05_solve_free_gs       (future: nonlinear physics)
    ↓
07_report / analysis

Key architectural rule:
-----------------------
Scripts DO NOT:
  • assemble PDE operators by hand
  • discretize equations
  • evaluate elliptic integrals directly

Scripts DO:
  • pass arrays to physics modules
  • interpret results
  • persist outputs to HDF5
  • decide *when* physics is called, not *how* it works


================================================================================
2) Inputs to physics modules (where they come from)
================================================================================

All physics inputs originate from either:
  • geometry modules (already persisted)
  • YAML profile configs
  • earlier physics outputs (ψ, fields)

Typical HDF5 reads before calling physics:
-------------------------------------------
From geometry:
  /grid/R
  /grid/Z
  /grid/RR
  /grid/ZZ
  /device/vessel_boundary
  /device/coil_greens/psi_per_amp

From target:
  /target/boundary
  /target/psi_boundary   (if precomputed)

From earlier physics:
  /equilibrium/psi_vac   (if stored separately)
  /equilibrium/psi       (previous iteration)

Rule:
-----
Physics modules never read HDF5 themselves.
Scripts *extract arrays* and pass them explicitly.


================================================================================
3) Vacuum field usage pattern (greens.py)
================================================================================

When it is used:
----------------
- Indirectly in 01_build_device.py
- Direct calls only from geometry/coils.py

Script-level contract:
----------------------
Scripts:
  • NEVER call psi_from_filament_loop directly
  • Treat coil Green’s functions as immutable geometry artifacts

Typical workflow:
-----------------
1) geometry/coils.compute_coil_psi_greens(...)
2) write /device/coil_greens/psi_per_amp to results.h5
3) later scripts read this tensor and linearly combine it

Linear recombination pattern:
-----------------------------
psi_vac = sum_k I_k * G_psi[k,:,:]

This linear structure is critical for:
  • optimization
  • fast re-evaluation
  • free-boundary iteration later

Important invariant:
--------------------
The ordering of coils in G_psi must match:
  /device/coils/names
  /device/coils/I_pf


================================================================================
4) Fixed-boundary GS usage (gs_solve_fixed.py)
================================================================================

When it is used:
----------------
- Script: 03_solve_fixed_gs.py
- This is the *first* nonlinear-looking physics step, but numerically linear

Script responsibilities BEFORE calling solver:
------------------------------------------------
1) Validate prerequisites via schema:
     - device
     - target
2) Load geometry:
     - R, Z, RR, ZZ
     - target LCFS boundary
3) Load vacuum field:
     - ψ_vac (computed from coil Green’s functions + currents)
4) Parse profile config (YAML)

Calling pattern:
----------------
psi_total = solve_fixed_boundary(
    R, Z, RR, ZZ,
    psi_vac,
    plasma_boundary,
    profile_cfg
)

What the solver assumes:
------------------------
- Grid is uniform and rectangular
- Boundary polyline is closed and well-behaved
- ψ_vac is defined on the full grid
- Profile config is physically reasonable

What the solver does NOT do:
----------------------------
- Does not write files
- Does not validate schema
- Does not plot
- Does not iterate nonlinearly

Script responsibilities AFTER solver:
-------------------------------------
1) Write ψ to HDF5:
     /equilibrium/psi
2) Optionally write ψ_vac separately
3) Validate schema stage "fixed_eq"
4) Optionally compute diagnostics (derived.py)


================================================================================
5) Operator reuse pattern (gs_operator.py)
================================================================================

When used:
----------
- Internally by gs_solve_fixed.py

Script-level implications:
--------------------------
- Scripts do NOT rebuild operators themselves
- Operator construction cost is hidden inside solver

Design intent:
--------------
Later, if nonlinear or free-boundary solvers require repeated solves:
  • operator reuse / caching can be added here
  • scripts remain unchanged

This keeps solver evolution local to physics layer.


================================================================================
6) Current profiles in practice (gs_profiles.py)
================================================================================

When profiles are evaluated:
-----------------------------
- During fixed-boundary solve
- Possibly repeatedly in future nonlinear iterations

Script responsibilities:
------------------------
- Provide profile configuration via YAML
- Decide which profile family to use
- Store profile parameters for provenance

Typical HDF5 persistence:
-------------------------
/profiles/config            (YAML snapshot or dict-as-attrs)
/profiles/description       (human-readable string)

Important usage rule:
---------------------
Scripts should treat profile choice as *input physics*, not solver logic.

If a new profile is needed:
  • add it to gs_profiles.py
  • do NOT hardcode profile logic in scripts


================================================================================
7) Field reconstruction usage (fields.py)
================================================================================

When it is used:
----------------
- After ψ has been computed
- Never inside the GS solve loop

Typical script usage:
---------------------
B_R, B_Z = compute_poloidal_field(R, Z, psi)

Persistence:
------------
/fields/B_R
/fields/B_Z

Use cases:
----------
- Plotting flux surfaces and field lines
- Diagnostics
- Derived quantity calculations

Important constraint:
---------------------
Field computation is *diagnostic*.
It must not feed back into the equilibrium solve at this stage.


================================================================================
8) Derived quantities usage (derived.py)
================================================================================

When it is used:
----------------
- After a successful GS solve
- In analysis, reporting, and optimization constraints

Typical quantities computed:
-----------------------------
- Magnetic axis (R0, Z0)
- ψ_axis, ψ_boundary
- Flux surface contours

Typical persistence:
--------------------
/derived/magnetic_axis/R
/derived/magnetic_axis/Z
/derived/psi_axis
/derived/psi_boundary

Design intent:
--------------
Derived quantities:
  • inform interpretation
  • guide optimization
  • do NOT alter equilibrium unless explicitly fed back later


================================================================================
9) How physics outputs become inputs again
================================================================================

Critical feedback loops (current + future):
--------------------------------------------
- ψ → derived quantities → optimization objective
- ψ → field → clearance / shaping constraints
- ψ → profile normalization → next GS solve (future)

Current status:
---------------
- Only one-way flow implemented
- No nonlinear iteration yet

Design constraint:
------------------
Any future feedback loop MUST be explicit at the script level.
Physics modules must remain single-purpose.


================================================================================
10) What scripts must NOT do (physics edition)
================================================================================

❌ Solve PDEs inline
❌ Modify ψ arrays in-place and continue silently
❌ Mix geometry construction with physics solving
❌ Bypass physics modules for “quick fixes”
❌ Hide iteration logic inside physics functions without documentation


================================================================================
11) Ready-for-free-boundary checklist
================================================================================

The current Pass B usage already prepares for free-boundary GS:

- [x] Vacuum Green’s functions precomputed
- [x] ψ_vac linear in coil currents
- [x] GS operator isolated
- [x] Profile evaluation modular
- [x] Solver stateless

What scripts will need to add later:
------------------------------------
- Outer iteration loop
- Boundary update logic
- Convergence criteria
- Optional relaxation

Physics modules will *extend*, not break.


================================================================================
Bottom line (Pass B)
================================================================================

From a script author’s perspective, the physics layer is:

  • a set of **pure numerical engines**
  • called at well-defined pipeline stages
  • fed entirely by geometry + config
  • returning arrays that become new workflow artifacts

If scripts follow the contracts described here:
  • physics remains debuggable
  • free-boundary extensions remain feasible
  • the workflow stays modular and comprehensible

This is exactly the structure you want before adding complexity.
