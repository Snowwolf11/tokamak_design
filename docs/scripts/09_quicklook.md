docs/scripts/09_quicklook.md
===========================
Script: scripts/09_quicklook.py

Date: 2026-01-16


Purpose
-------
`09_quicklook.py` generates a standardized set of diagnostic plots (“quicklook”)
for a run by reading `results.h5` and calling `tokdesign.viz.*` plotting helpers.

It is an *orchestrator*:
  • reads all data ONLY via `tokdesign.io.h5` (single I/O layer)
  • does not compute physics/optimization (except trivial derived visuals like |B|)
  • calls plotting functions and saves figures into `run_dir/figures/` in a
    consistent subdirectory structure

It is designed to work at different pipeline maturity levels:
  • after stage 01 it can still plot device geometry
  • after stage 02 it overlays the target LCFS
  • after stage 04 it adds coil-fit diagnostics and vacuum flux comparisons
  • after stage 03 (and later) it adds equilibrium/fields/profiles/derived plots


Output structure (filesystem)
-----------------------------
Writes into:

run_dir/figures/
  01_device/       device geometry overview
  02_target/       target boundary overlay
  03_greens/       coil Green’s function maps (subset)
  04_coil_fit/     coil-fit diagnostics + vacuum flux surfaces
  10_equilibrium/  ψ map and scalar equilibrium maps (p, F, jphi)
  11_fields/       |Bp|, |B|, midplane cuts, optional 3D field lines
  12_profiles/     binned profiles vs normalized ψ
  13_derived/      derived scalars summary, q-profile plot

Files saved as:
  <name>.<fmt> for each format in --formats (e.g. png, pdf)

It creates subdirectories as needed.


Inputs (CLI)
------------
Required:
  --run-dir <path>
      Run directory containing results.h5

Optional output controls:
  --formats <fmt...>
      Default: ["png"]
      Examples: png pdf svg

  --dpi <int>
      Default: 160
      Applies to raster outputs (png).

Greens controls:
  --coils <name...>
      Specific coil names to plot greens for (default: first few)
  --greens-max <int>
      Default: 4 (max number of greens plots)
  --greens-levels <int>
      Default: 30 (contour levels)

Fieldline controls (optional):
  --fieldlines
      If set, also plot exemplary 3D field lines (requires BR,BZ,Bphi in HDF5)
  --fieldline-steps <int>
      Default: 2500 integration steps
  --fieldline-ds <float>
      Default: 0.01 step size (pseudo arclength step)


Inputs (HDF5 datasets read)
---------------------------
This script is intentionally defensive: it checks whether each path exists
before reading it.

Always read (required for *any* run that passed init):
  /meta/schema_version
  /meta/run_id

Device (if present):
  /device/vessel_boundary
  /device/coils/names
  /device/coils/centers
  /device/coils/radii          (NOTE: this expects “radii”, while other docs/stages
                                sometimes call this “a”; see gotchas)
  /device/coils/I_pf
  /device/coils/I_max

Target (if present):
  /target/boundary
  /target/psi_boundary

Grid (if present):
  /grid/R
  /grid/Z
  /grid/RR

Greens (if present):
  /device/coil_greens/psi_per_amp

Stage-04 coil fit results (optional):
  /optimization/fit_results/boundary_points
  /optimization/fit_results/psi_boundary_fit
  /optimization/fit_results/clamped
  /optimization/fit_results/contour_rms
  /optimization/fit_results/psi_ref

Equilibrium and related (optional):
  /equilibrium/psi
  /equilibrium/psi_axis
  /equilibrium/psi_lcfs         (fallback to target psi_boundary if missing)
  /equilibrium/p_psi
  /equilibrium/F_psi
  /equilibrium/jphi
  /equilibrium/plasma_mask

Fields (optional):
  /fields/BR
  /fields/BZ
  /fields/Bphi

Derived (optional):
  /derived/Ip
  /derived/beta_p
  /derived/li
  /derived/kappa
  /derived/delta
  /derived/q_profile


Outputs (plots) by section
--------------------------

01_device: device geometry
--------------------------
Always attempted.
Calls:
  tokdesign.viz.plot_device.plot_device_geometry(...)
Shows:
  • vessel boundary
  • PF coil positions/sizes
  • coil currents (if present)

Saved as:
  figures/01_device/device_geometry.<fmt>

02_target: target overlay
-------------------------
Always attempted (even if target missing; plotter should handle None).
Calls:
  tokdesign.viz.plot_target.plot_target_overlay(...)
Shows:
  • device geometry
  • target LCFS boundary overlay
  • optional shape annotation
  • psi_lcfs value annotation (uses equilibrium psi_lcfs if available else psi_target)

Saved as:
  figures/02_target/target_overlay.<fmt>

03_greens: coil Green’s functions (optional)
--------------------------------------------
Only if:
  G_psi, R, Z, coil_names exist.

Calls:
  tokdesign.viz.plot_greens.plot_coil_greens(...)

Produces up to:
  --greens-max plots
For either:
  • user-selected coils via --coils, or
  • the first few coils

Saved as:
  figures/03_greens/greens_<coilname>.<fmt>

04_coil_fit: diagnostics + vacuum flux surfaces (optional)
----------------------------------------------------------
If fit results exist:
  boundary_points + psi_boundary_fit
then:
  • plot_boundary_psi_vs_angle
  • plot_boundary_dpsi
and if also coil currents and limits exist:
  • plot_currents_vs_limits

Additionally, if coil greens + coil currents + grid exist:
  • compute_psi_vacuum(G_psi, I_pf)   (simple tensor contraction)
  • plot_vacuum_psi_map
  • plot_vacuum_flux_surfaces_compare

The comparison plot can overlay:
  • vacuum flux surfaces (from fitted currents)
  • and equilibrium ψ contours (if /equilibrium/psi exists)
It chooses a meaningful reference contour level:
  • psi_ref if present
  • else mean(psi_boundary_fit)

Saved as:
  figures/04_coil_fit/boundary_psi_vs_theta.<fmt>
  figures/04_coil_fit/delta_psi_vs_theta.<fmt>
  figures/04_coil_fit/currents_vs_limits.<fmt>
  figures/04_coil_fit/vacuum_psi_map.<fmt>
  figures/04_coil_fit/vacuum_flux_surfaces_compare.<fmt>

10_equilibrium / 11_fields / 12_profiles / 13_derived (optional)
----------------------------------------------------------------
These sections run only if:
  R, Z, and /equilibrium/psi exist.

10_equilibrium:
  • plot_psi_map (ψ contours and LCFS overlay)
  • plot_scalar_map for p_psi, F_psi, jphi if present

11_fields:
  • computes Bp = sqrt(BR^2 + BZ^2) if BR,BZ present
  • computes |B| if BR,BZ,Bphi present
  • plot_scalar_map for Bp and |B|
  • plot_midplane_cuts for multiple quantities at Z=0
  • optional 3D fieldline plot (if --fieldlines and all field components exist)

12_profiles:
  If psi_axis, psi_lcfs, and plasma_mask exist:
    • compute psin = clip((psi-psi_axis)/(psi_lcfs-psi_axis), 0..1)
    • optionally compute Bp and |B|
    • plot_profiles_vs_psin with binning (nbins=45)

13_derived:
  • a text-only derived summary figure for scalars (Ip, beta_p, li, kappa, delta)
  • q-profile plot if /derived/q_profile exists

Saved as:
  figures/10_equilibrium/psi_map.<fmt>
  figures/10_equilibrium/p_map.<fmt>        (optional)
  figures/10_equilibrium/F_map.<fmt>        (optional)
  figures/10_equilibrium/jphi_map.<fmt>     (optional)
  figures/11_fields/Bp_map.<fmt>            (optional)
  figures/11_fields/Bmag_map.<fmt>          (optional)
  figures/11_fields/midplane_cuts.<fmt>
  figures/11_fields/fieldlines_3d.<fmt>     (optional)
  figures/12_profiles/profiles_vs_psin.<fmt> (optional)
  figures/13_derived/derived_summary.<fmt>
  figures/13_derived/q_profile.<fmt>        (optional)


Core internal helpers
---------------------

_decode_strings(arr) -> List[str]
  - Converts numpy arrays of bytes/np.bytes_ into utf-8 strings.
  - Used for coil name decoding from HDF5.

save_figure(fig, out_base, formats, dpi)
  - Ensures directory exists, then saves the same figure in multiple formats.

_maybe(h5, path) -> bool
  - Convenience existence check: `path in h5`.


Algorithmic details worth noting
--------------------------------

A) Defensive “partial-run” behavior
-----------------------------------
Quicklook does not assume you ran the full workflow.
It conditionally plots sections based on dataset availability.

This is extremely useful during development:
  • you can run quicklook after stage 01 and still get meaningful output

B) Vacuum ψ computation used for plots
--------------------------------------
When greens and currents exist, quicklook computes:

  psi_vac = compute_psi_vacuum(G_psi, I_pf)

This is not a solver; it is a linear contraction (O(Nc×NR×NZ)).
It supports “before/after fit” visualization and vacuum flux surface plots.

C) Normalized flux psin construction
------------------------------------
If psi_axis and psi_lcfs exist:
  psin = (psi - psi_axis) / (psi_lcfs - psi_axis)
clipped to [0,1].

This is used only for profiling/binned diagnostics.
It assumes:
  • monotonic ψ from axis to LCFS
  • consistent sign convention
It includes a simple check that denom != 0.

D) Fieldline seed selection heuristic
-------------------------------------
If fieldlines are requested and equilibrium data exist, it tries to select
reasonable starting points on the midplane using:
  • plasma_mask at Z≈0
  • psin values near [0.2, 0.4, 0.6, 0.8]
It falls back to a single seed at (R[mid], Z=0) if it can’t find good seeds.

This is a pragmatic heuristic aimed at producing “something sensible” without
requiring extra user input.


Dependencies
------------
Stdlib:
  argparse, pathlib, typing

Third-party:
  numpy
  matplotlib

Internal:
  tokdesign.io.h5:
    open_h5, h5_read_array, h5_read_scalar
  tokdesign.viz.style:
    apply_mpl_defaults
  tokdesign.viz.plot_device:
    plot_device_geometry
  tokdesign.viz.plot_target:
    plot_target_overlay
  tokdesign.viz.plot_greens:
    plot_coil_greens
  tokdesign.viz.plot_equilibrium:
    plot_psi_map, plot_scalar_map, plot_midplane_cuts
  tokdesign.viz.plot_profiles:
    plot_profiles_vs_psin
  tokdesign.viz.plot_fieldlines:
    plot_fieldlines_3d
  tokdesign.viz.plot_coil_fit:
    plot_boundary_psi_vs_angle, plot_boundary_dpsi, plot_currents_vs_limits,
    compute_psi_vacuum, plot_vacuum_flux_surfaces_compare, plot_vacuum_psi_map


Gotchas / mismatches to watch
-----------------------------
1) Coil “radii” dataset name
   - quicklook reads `/device/coils/radii`
   - earlier geometry docs and schema discussions sometimes refer to coil size as
     “a” (PFCoil.a) and/or `/device/coils/a`.
   If your pipeline writes `/device/coils/a` instead, quicklook will not display
   coil sizes unless you:
     • also write `/device/coils/radii`, or
     • update quicklook to read `/device/coils/a` as fallback.

2) Equilibrium scalar path names
   - quicklook looks for:
       /equilibrium/psi_axis
       /equilibrium/psi_lcfs
     but stage-03 documentation earlier treated these as “optional”.
   If 03_solve_fixed_gs doesn’t write them, quicklook will still work but:
     • psin-based plots (profiles) may be skipped
     • psi_lcfs falls back to target psi_boundary.

3) Plasma mask expected as a dataset
   - quicklook expects /equilibrium/plasma_mask and casts to bool.
   If your GS solver does not store it, it will still plot ψ maps, but it will
   not mask scalar fields (p, F, jphi) to the plasma region.

4) “wrote_any” variable is unused
   - `wrote_any` is set during greens plotting but not used later.
   Harmless; could be removed or used for summary messaging.


Practical usage patterns
------------------------
A) Default quicklook (png only):
  python scripts/09_quicklook.py --run-dir data/runs/<RUN_ID>

B) Save both png and pdf:
  python scripts/09_quicklook.py --run-dir data/runs/<RUN_ID> --formats png pdf

C) Plot more coil greens:
  python scripts/09_quicklook.py --run-dir ... --greens-max 8

D) Include 3D fieldlines (requires /fields/*):
  python scripts/09_quicklook.py --run-dir ... --fieldlines

E) Specify particular coils for greens:
  python scripts/09_quicklook.py --run-dir ... --coils PF1 PF2 CS


One-line summary
----------------
`09_quicklook.py` is the standardized plotting dashboard:
it reads whatever exists in results.h5 and writes a consistent set of figures
into run_dir/figures/ subfolders, scaling gracefully with pipeline progress.
