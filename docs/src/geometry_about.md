docs/src/geometry_about.md
=========================

16.01.26

Pass A: Module contract for `src/.../geometry/`

Date: 2026-01-17

Scope
-----
This document describes the `src/.../geometry` subfolder in detail:
idea/philosophy, included modules, public API, inputs/outputs, math/geometry,
dataflow touchpoints, and dependencies.

Folder philosophy (“geometry”)
------------------------------
The geometry layer defines *shapes and spatial representations* used throughout
the tokamak design workflow.

Key rule:
  • Geometry contains NO equilibrium solving, optimization, or “physics state”.
  • It provides reusable representations and queries:
      - computational grids (R-Z)
      - boundary polylines (LCFS and vessel)
      - coil geometry bookkeeping (PF coils + solenoid)
      - geometric metrics and helper utilities

The geometry layer is intentionally lightweight and depends mostly on:
  • numpy
  • Python stdlib (dataclasses, typing)

It may *call into physics* only at clearly defined seams (e.g. “single coil Green’s
function”), and even then it should avoid deep coupling by importing lazily.


Included modules
----------------
- grids.py
- plasma_boundary.py
- coils.py
- vessel.py


================================================================================
grids.py — R–Z grid utilities
================================================================================

Purpose
-------
Provide a consistent way to build and reason about the 2D computational grid used by:
  • Grad–Shafranov solvers
  • coil/vacuum field Green’s function precomputations
  • postprocessing (contours, derivatives, masks)

Core conventions
----------------
- 1D coordinate arrays:
    R: shape (NR,)
    Z: shape (NZ,)
- 2D mesh arrays:
    RR, ZZ: shape (NZ, NR)
  Ordering is (Z index first, then R index).
  This matches “image-like” storage and typical contour plotting.

Units
-----
All lengths are SI meters [m].

Public API
----------
1) make_rz_grid(R_min, R_max, Z_min, Z_max, NR, NZ, *, endpoint=True)
   -> (R, Z, RR, ZZ)

   Inputs:
     R_min, R_max [m], Z_min, Z_max [m], NR, NZ >= 2
   Outputs:
     R (NR,), Z (NZ,), RR (NZ,NR), ZZ (NZ,NR)

   Implementation notes:
     - uses np.linspace for uniform grid
     - uses np.meshgrid(indexing="xy") so RR varies along axis=1 and ZZ along axis=0

   Important guard:
     - rejects R_min <= 0.0 with a ValueError, since axisymmetric tokamak formulas
       typically assume R > 0 (avoid singularities at R=0)

2) grid_spacing(R, Z) -> (dR, dZ)

   Purpose:
     - verify that R and Z are strictly increasing and approximately uniform
     - return scalar spacings (v1 assumes uniform grids)

   Validations:
     - R and Z must be 1D, length >= 2
     - strictly increasing
     - uniformity checked with np.allclose at very tight rtol=1e-10

Dependencies
------------
- numpy
- stdlib typing

Dataflow touchpoints
--------------------
This module doesn’t write files by itself. In the workflow, its outputs typically
end up in results.h5 under:
  /grid/R, /grid/Z, /grid/RR, /grid/ZZ
(the exact writing is done by scripts using io/h5.py)

Design notes / extension points
-------------------------------
- If later you want non-uniform grids, grid_spacing() is designed to be replaced
  or complemented by a “nonuniform” function.
- The strict uniformity tolerance is intentionally conservative; if you generate
  grids by arithmetic or external sources you may need to relax it.


================================================================================
plasma_boundary.py — LCFS boundary geometry utilities
================================================================================

Purpose
-------
Provide simple, reusable tools to:
  • generate a Miller-parameterized plasma boundary polyline
  • compute the enclosed area of a closed boundary polyline
  • estimate elongation κ and triangularity δ from a boundary polyline

Core conventions
----------------
- A boundary is a polyline array of shape (N, 2):
    poly[:,0] = R [m]
    poly[:,1] = Z [m]
- Generated boundaries are CLOSED:
    poly[0] == poly[-1]

This module is purely geometric:
  • it does not compute equilibria or solve Grad–Shafranov
  • it defines target shapes and derived shape metrics

Math: Miller parameterization
-----------------------------
Given major radius R0, minor radius a, elongation κ, triangularity δ:

  R(θ) = R0 + a * cos( θ + δ * sin(θ) )
  Z(θ) = κ * a * sin(θ)

Interpretation:
  • κ controls vertical stretching
  • δ > 0 shifts the top/bottom inward (“D-shape”)

Public API
----------
1) miller_boundary(R0, a, kappa, delta, npts=400, *, closed=True) -> np.ndarray

   Inputs:
     R0 > 0 [m], a > 0 [m], kappa > 0, delta (dimensionless), npts >= 32
   Output:
     boundary polyline of shape (npts+1, 2) if closed, else (npts, 2)

   Implementation notes:
     - theta is uniform on [0, 2π) (endpoint=False)
     - closure performed by ensure_closed_polyline()

2) boundary_area(poly) -> float

   Uses shoelace formula on a closed polygon:
     area = 0.5 * |Σ x_i y_{i+1} - y_i x_{i+1}|
   Returns positive area [m^2]

   Inputs:
     poly shape (N,2), will be closed automatically
   Output:
     float area [m^2]

3) boundary_kappa_delta(poly) -> (kappa, delta)

   Computes geometric estimates using “engineering” definitions:
     R_in  = min(R), R_out = max(R)
     a     = (R_out - R_in)/2
     R0    = (R_out + R_in)/2

     Z_max = max(Z), Z_min = min(Z)
     κ     = (Z_max - Z_min) / (2a)

   Triangularity:
     R_top = mean R of points near Z_max (tolerance-based)
     R_bot = mean R of points near Z_min
     δ_top = (R0 - R_top)/a
     δ_bot = (R0 - R_bot)/a
     δ     = (δ_top + δ_bot)/2

   Notes/limitations:
     - This is a discrete-point estimate, not a smooth fit.
     - Diverted/X-point shapes can make δ ambiguous; v1 assumes single-valued D-shapes.

Helper utilities
----------------
- ensure_closed_polyline(poly) -> poly_closed
- _validate_polyline(poly) sanity checks:
    * shape (N,2)
    * closed boundary should have >= 4 points (incl closure)
    * finite values
    * R > 0 everywhere (tokamak R-Z convention)

Dependencies
------------
- numpy
- stdlib typing

Dataflow touchpoints
--------------------
This module doesn’t write files. In the workflow, outputs typically map to:
  /target/boundary (when used to generate a target LCFS)
and derived metrics may be stored under /derived or /analysis (depending on design).


================================================================================
coils.py — PF coil geometry + vacuum psi assembly interface
================================================================================

Purpose
-------
Provide:
  • a lightweight PFCoil dataclass (filamentary circular loop approximation)
  • config parsing for coils from baseline_device.yaml-like dicts
  • array extraction utilities (centers, currents, limits)
  • current-update helper that returns new coil objects
  • a seam for coil Green’s functions:
        compute_coil_psi_greens(...) -> G_psi
        psi_from_coils(G_psi, I) -> psi_vac

Separation of concerns (important)
----------------------------------
This module defines coil geometry + bookkeeping.
The actual physics for psi(R,Z) from a circular current loop is explicitly NOT here.
Instead, it is delegated to:
  tokdesign.physics.greens.psi_from_filament_loop(...)

To avoid circular imports and keep geometry lightweight, the import happens inside
compute_coil_psi_greens() (lazy import).

Core conventions
----------------
- Positions Rc, Zc in meters [m]
- Currents I and limits I_max in amperes [A]
- Grid arrays RR, ZZ have shape (NZ, NR)
- Green’s function tensor:
    G_psi has shape (Nc, NZ, NR)
  such that:
    psi_vac(R,Z) = Σ_k G_psi[k,:,:] * I_k

Public API
----------
Dataclass:
1) PFCoil (frozen)
   Attributes:
     name: str
     Rc, Zc: float [m]  (coil center)
     a: float [m]       (“size proxy” for plotting/clearance; not used in v1 physics)
     I: float [A]
     I_max: float [A]   (absolute max magnitude)

Config parsing:
2) coils_from_config(cfg: dict, *, include_solenoid=True) -> List[PFCoil]

   Expected structure (conceptual):
     pf_coils:
       coils:
         - name, Rc, Zc, a, I_init, I_max
     central_solenoid:
       enabled: true/false
       Rc, Zc, a, I_init, I_max

   Behavior:
     - builds PFCoil objects for pf_coils.coils
     - optionally appends a solenoid coil named "CS" if enabled
     - raises if no coils found
     - validates coils via _validate_coils()

Array helpers:
3) coil_centers(coils) -> np.ndarray shape (Nc,2)  [[Rc,Zc], ...]
4) coil_currents(coils) -> np.ndarray shape (Nc,)
5) coil_current_limits(coils) -> np.ndarray shape (Nc,)

Mutation helper:
6) set_coil_currents(coils, I) -> List[PFCoil]
   - returns NEW PFCoil objects with updated I values (immutability-friendly)
   - validates I shape matches number of coils

Green’s interface + vacuum psi assembly:
7) compute_coil_psi_greens(coils, RR, ZZ, *, method="analytic_elliptic") -> np.ndarray

   Inputs:
     coils: sequence of PFCoil (geometry used; current values irrelevant for per-amp)
     RR, ZZ: mesh grids (NZ,NR)
     method: passed through to psi_from_filament_loop

   Output:
     G_psi: (Nc, NZ, NR) psi per ampere for each coil

   Implementation:
     - loops coils and calls psi_from_filament_loop(RR, ZZ, Rc, Zc, I=1.0, method=...)

   Dependency seam:
     requires tokdesign.physics.greens.psi_from_filament_loop to exist

8) psi_from_coils(G_psi, I) -> psi_vac (NZ,NR)
   - contracts over coil index: tensordot(I, G_psi, axes=(0,0))

Validation behavior
-------------------
- _validate_coils checks:
    * unique coil names
    * finite values
    * Rc > 0
    * a >= 0
    * I_max > 0
- compute_coil_psi_greens validates RR and ZZ are 2D and same shape

Dependencies
------------
- numpy
- stdlib: dataclasses, typing
- physics.greens (lazy import inside function)

Dataflow touchpoints
--------------------
This module itself doesn’t perform file I/O, but its outputs are intended to be
stored in results.h5 by stage scripts, typically under paths like:
  /device/coils/names
  /device/coils/centers
  /device/coils/radii (or a proxy like "a")
  /device/coils/I_pf
  /device/coils/I_max
  /device/coil_greens/psi_per_amp   (this is explicitly mentioned in schema v0.1)

Note: The exact dataset names (centers vs Rc/Zc arrays, radii vs a proxy) should
be kept consistent with schema.py to preserve the “HDF5 is the API” contract.


================================================================================
vessel.py — Vessel / wall boundary utilities
================================================================================

Purpose
-------
Represent the vacuum vessel (or first wall) boundary in R–Z as a closed polyline
and provide geometric queries used by the workflow:
  • point-in-vessel tests (inside/outside mask)
  • minimum distance from points to vessel boundary (clearance constraints)
  • parametric vessel generator (ellipse) for early studies
  • config parsing to build vessel boundary from baseline_device.yaml-like dicts

Core conventions
----------------
- Vessel boundary is a polyline array of shape (N,2):
    [:,0] = R [m]
    [:,1] = Z [m]
- Vessel polylines are assumed CLOSED for inside/outside tests.
  If not closed, the module closes them automatically.

Public API
----------
1) load_vessel_from_config(cfg: Dict[str,Any]) -> np.ndarray

   Expected config shape (conceptual):
     vessel:
       representation: "parametric" or "polyline"
       parametric:
         R_center, Z_center, a_wall, b_wall, n_points
       polyline:
         points: [[R1,Z1],[R2,Z2],...]

   Behavior:
     - parametric -> ellipse_boundary(...)
     - polyline   -> uses given points
     - always closes (ensure_closed_polyline) and validates

2) ellipse_boundary(R_center, Z_center, a_wall, b_wall, *, n_points=400) -> np.ndarray
   - returns closed ellipse polyline (n_points+1, 2)
   - validates n_points >= 16 and positive semi-axes

3) ensure_closed_polyline(poly) -> poly_closed
   - appends first point if needed (shape (N,2), N>=3)

4) point_in_polygon(poly, pts) -> inside (M,)
   - vectorized ray casting method
   - pts shape (M,2)
   - returns boolean mask
   - note: points exactly on boundary may be classified either way (acceptable for masks)

5) min_distance_to_polyline(poly, pts) -> dmin (M,)
   - computes min Euclidean distance from each point to any segment
   - implementation:
       for each segment a->b:
         project p onto segment, clamp t in [0,1]
         compute closest point and distance
         take min over segments
   - ensures closed polyline to include last segment

Internal validation:
- _validate_polyline(poly):
    * shape (N,2)
    * closed boundary >= 4 points (incl closure)
    * finite values
    * all R > 0 (tokamak convention)

Dependencies
------------
- numpy
- stdlib typing

Dataflow touchpoints
--------------------
As with other geometry modules, this module doesn’t write files directly. Its
outputs are intended to be persisted to results.h5 by scripts, typically:
  /device/vessel_boundary

It is also used for:
  • clearance constraints (LCFS-to-wall distance)
  • mask building for solvers (if you later use vessel as computational domain)


================================================================================
Cross-module contract summary (geometry folder)
================================================================================

What other layers should rely on from geometry:
- Standard grid representation:
    R (NR,), Z (NZ,), RR/ZZ (NZ,NR)
- Standard polyline representation for boundaries:
    poly (N,2) with columns (R,Z), closed when required
- Coil representation:
    PFCoil list + deterministic array extraction helpers
- Vessel queries:
    point_in_polygon and min_distance_to_polyline for constraints/masks

Geometry should remain:
- deterministic (given inputs, outputs are pure)
- lightweight (no heavy dependencies)
- stable in its shapes/conventions (so downstream code doesn’t break)


================================================================================
Notes / Action items emerging from Pass A
================================================================================
1) Duplication of ensure_closed_polyline and _validate_polyline:
   - plasma_boundary.py and vessel.py both define very similar helpers.
   - This is not “wrong”, but if you want to reduce drift, you may factor these
     helpers into a shared geometry/polyline_utils.py later.

2) coils.py: clarify naming of “a”:
   - PFCoil.a is described as a “size proxy” not used in physics (filament model).
   - In HDF5 schema, a dataset name like “radii” vs “a” should be consistent to
     avoid confusion.

3) compute_coil_psi_greens() has a clean physics seam:
   - good separation: geometry handles looping/assembly, physics handles formula.
   - keep it that way (avoid importing big physics modules at top-level).

4) grids.py rejects R_min <= 0 by ValueError:
   - good default for tokamaks; scripts/configs must respect this.
   - if you later want diagnostic/test grids including R=0, you may add a flag.
