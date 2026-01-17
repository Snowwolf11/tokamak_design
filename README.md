Tokamak Design Workflow
=======================

Overview
--------
This repository implements a staged, reproducible workflow for **tokamak design**,
starting from the *machine description* (geometry, coils, vessel) and progressing
towards *plasma equilibria*, *coil current optimization*, and *physics diagnostics*.

The long-term goal is ambitious and explicit:

  → **Design a tokamak from the machine outward to the plasma physics,
     within one coherent, inspectable workflow.**

What you see here is the **current state** of that vision:
  • geometry definition
  • target plasma shape specification
  • fixed-boundary Grad–Shafranov equilibria
  • PF coil current fitting
  • standardized diagnostics and visualization

The architecture is deliberately designed so that future extensions —
free-boundary equilibria, stability analysis, profile self-consistency,
advanced optimization — can be added without breaking existing stages.


===============================================================================
Core philosophy
===============================================================================

This project follows a few strict principles that shape *everything*:

1) Staged workflow with explicit artifacts
------------------------------------------
Each step in the workflow is a **stage**.
Stages communicate only via **persistent artifacts**, not Python objects.

The canonical artifact is:
  • a run directory
  • containing a single authoritative data file: `results.h5`

Once a stage completes, its outputs are written to HDF5 and treated as immutable
inputs for downstream stages.

This enables:
  • resumability
  • reproducibility
  • debuggability
  • independent re-running of stages


2) Separation of concerns
-------------------------
Responsibilities are cleanly separated:

- `scripts/`
    CLI entry points and orchestration only
    (no physics, no geometry algorithms)

- `src/`
    reusable, testable implementation code:
      • io        → paths, logging, HDF5, schema
      • geometry  → grids, coils, vessel, boundaries
      • physics   → Green’s functions, GS solvers, fields
      • optimization → PF current fitting
      • viz       → plotting utilities

Scripts *call into* src modules.
src modules never read YAML, parse CLI args, or touch the filesystem directly.


3) HDF5 is the API
------------------
The layout of `results.h5` is a **formal interface contract**.

Paths like:
  /grid/R
  /device/coil_greens/psi_per_amp
  /target/boundary
  /equilibrium/psi

are treated exactly like function signatures in a software library.

If these change:
  → the schema version must change.

This mindset keeps the workflow stable as complexity grows.


===============================================================================
Repository structure
===============================================================================

Top-level layout:

  configs/        YAML input files
  scripts/        workflow stages (CLI)
  src/            core implementation
  docs/           detailed documentation
  data/runs/      auto-generated run directories

Each run directory contains:
  inputs/         archived YAML configs (provenance)
  figures/        plots and reports
  run.log         full execution log
  results.h5     canonical data store


===============================================================================
The workflow stages
===============================================================================

Stage 00 — Initialize a run
---------------------------
Script: `00_init_run.py`

This stage creates a new run directory and initializes `results.h5`.

Responsibilities:
  • generate a unique run ID
  • create the standard run directory structure
  • archive YAML input configs into `run_dir/inputs/`
  • write metadata to `/meta/*` in HDF5
  • initialize `/history` for provenance tracking
  • validate schema stage "init"

This stage guarantees that every run is self-contained and reproducible.


Stage 01 — Build the device
---------------------------
Script: `01_build_device.py`

This stage defines everything that is **static and plasma-independent**:

  • R–Z computational grid
  • vacuum vessel boundary
  • PF coil geometry and current limits
  • coil → ψ Green’s functions

These artifacts form the *machine contract*.
They are computed once and treated as immutable thereafter.

This stage is computationally heavier (especially Green’s functions) by design,
so downstream stages can remain fast.


Stage 02 — Define the target plasma boundary
--------------------------------------------
Script: `02_target_boundary.py`

This stage encodes *what we want the plasma to look like*:

  • target LCFS boundary (typically via Miller parameterization)
  • prescribed boundary flux value
  • global scalar targets (e.g. total plasma current)

No equilibrium is solved here.
This stage defines constraints and objectives, not solutions.


Stage 03 — Fixed-boundary Grad–Shafranov equilibrium
----------------------------------------------------
Script: `03_solve_fixed_gs.py`

This is the first true physics solve.

Inside the prescribed LCFS:
  • the Grad–Shafranov equation is discretized using finite differences
  • a sparse linear system is assembled
  • Dirichlet boundary conditions are enforced on the LCFS
  • a direct sparse solver computes ψ on the grid

This produces a **fixed-boundary equilibrium**, suitable as:
  • a baseline equilibrium
  • a regression test
  • a starting point for future free-boundary iteration


Stage 04 — Fit PF coil currents
-------------------------------
Script: `04_fit_pf_currents.py`

This stage fits PF coil currents so that the **vacuum poloidal flux**
matches the target boundary constraints.

Key features:
  • linear or ridge-regularized least squares
  • optional scaling by coil current limits
  • enforcement of engineering bounds
  • detailed fit diagnostics written to HDF5
  • history snapshots of overwritten currents

The result is an updated set of canonical PF currents at:
  /device/coils/I_pf

These currents are intended to be good initial guesses for later
free-boundary equilibrium calculations.


Stage 09 — Quicklook diagnostics
--------------------------------
Script: `09_quicklook.py`

This stage generates a standardized set of plots by *reading only* `results.h5`.

It is intentionally robust to partial runs:
  • after stage 01 → device geometry plots
  • after stage 02 → target overlay
  • after stage 03 → equilibrium ψ and fields
  • after stage 04 → coil-fit diagnostics

Plots are written to:
  run_dir/figures/
in a structured subdirectory layout.


Workflow driver
---------------
Script: `run_workflow.py`

This is a convenience wrapper that runs stages in sequence.

Features:
  • one-command execution up to a chosen stage
  • automatic discovery of the new run directory
  • preference for archived configs in `run_dir/inputs/`
  • optional automatic quicklook at the end

Each stage remains fully usable on its own;
the driver only orchestrates them.


===============================================================================
Physics and numerical approach (current status)
===============================================================================

Implemented physics:
  • analytic Green’s functions for filamentary circular coils
  • vacuum field computation via linear superposition
  • fixed-boundary Grad–Shafranov solver
  • basic plasma current profile models
  • magnetic field reconstruction from ψ
  • derived equilibrium diagnostics

Numerical methods:
  • uniform-grid finite differences
  • sparse matrix assembly (CSR)
  • direct sparse linear solves
  • linear and ridge-regularized least squares
  • optional bound enforcement for optimization

The emphasis is on:
  • algorithmic clarity
  • inspectability
  • correctness over premature optimization


===============================================================================
Documentation
===============================================================================

This document provides the **big-picture overview**.

Much more detailed documentation lives in the `docs/` folder:

- `docs/src/`
    Pass A / Pass B documents for each `src` submodule:
      • design philosophy
      • physics background
      • numerical algorithms
      • data contracts

- `docs/scripts/`
    One document per script describing:
      • inputs and outputs
      • dataflow
      • numerical meaning
      • failure modes and assumptions

For onboarding new contributors, start with:
  → this document
  → then `docs/`
  → then the scripts in execution order


===============================================================================
Current status and future direction
===============================================================================

What you see here is **not the final state**, but a carefully designed foundation.

Planned extensions include:
  • free-boundary Grad–Shafranov equilibrium solvers
  • nonlinear profile iteration and self-consistency
  • X-point and separatrix handling
  • stability analysis (e.g. ideal MHD metrics)
  • tighter coupling between equilibrium, coils, and profiles
  • more advanced optimization objectives
  • expanded diagnostics and reporting

The architecture is intentionally built so that:
  • these features can be added incrementally
  • existing runs and stages remain valid
  • complexity grows without losing conceptual clarity


===============================================================================
Bottom line
===============================================================================

This repository is a **design workflow**, not just a solver.

Its goal is to:
  → start from a machine description
  → move through geometry and control
  → arrive at plasma equilibria and physics insight
  → all within one coherent, reproducible framework.

The current implementation provides a solid, explicit base on which
much more sophisticated tokamak design and physics capabilities can be built.
