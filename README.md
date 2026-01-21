# TOKAMAK DESIGN WORKFLOW

A staged, reproducible framework for tokamak design  
from machine geometry to plasma physics

---

## TABLE OF CONTENTS

1. Motivation and scope  
2. What this repository does (and does not do)  
3. Core design philosophy  
4. Repository structure  
5. The run concept and data model  
6. Workflow stages (scripts)  
7. Core implementation layers (src/)  
8. Typical usage patterns  
9. Extending the workflow  
10. Documentation and onboarding  
11. Current status and future roadmap  
12. Bottom line  

---

## 1. MOTIVATION AND SCOPE

Designing a tokamak is inherently a multi-layered problem:

- machine geometry (vessel, coils, grids)  
- control and actuation (PF currents)  
- plasma equilibrium physics  
- constraints, diagnostics, and iteration  

In many projects, these aspects are handled by loosely connected scripts,
monolithic solvers with hidden state, or manual, error-prone workflows.

This repository exists to provide a **SINGLE, COHERENT, INSPECTABLE WORKFLOW**
that takes you:

- from a machine description  
- through geometry and control  
- to plasma equilibria and physics diagnostics  

all within one consistent framework.

This is not a black-box solver.  
It is an instructional, research-grade workflow designed to grow over time.

The long-term goal is explicit:

> DESIGN A TOKAMAK FROM THE MACHINE UP TO THE PLASMA PHYSICS  
> IN ONE INTEGRATED WORKFLOW.

What you see here is the **CURRENT STATUS** of that vision, not its final form.

---

## 2. WHAT THIS REPOSITORY DOES (AND DOES NOT DO)

### WHAT IT DOES TODAY

- Defines tokamak geometry (grid, vessel, PF coils)  
- Precomputes analytic coil Green’s functions  
- Defines target plasma boundaries (LCFS)  
- Solves fixed-boundary Grad–Shafranov equilibria  
- Fits PF coil currents to target boundary constraints  
- Produces standardized diagnostic plots (“quicklook”)  
- Tracks provenance and history in a single data file  

### WHAT IT DOES NOT YET DO

- Free-boundary Grad–Shafranov equilibria  
- Self-consistent nonlinear profile iteration  
- X-point / separatrix equilibrium solving  
- Stability analysis (ideal MHD, etc.)  

These are **PLANNED EXTENSIONS**.  
The current architecture is deliberately designed so these can be added
incrementally without breaking existing stages.

---

## 3. CORE DESIGN PHILOSOPHY

### 3.1 STAGED WORKFLOW WITH EXPLICIT ARTIFACTS

The workflow is divided into **STAGES**.  
Each stage:

- has a single responsibility  
- reads well-defined inputs  
- writes well-defined outputs  
- validates its results structurally  

Stages **NEVER** pass Python objects to each other.

Instead, they communicate through:

- a run directory  
- a single canonical data file: `results.h5`  

This makes the workflow:

- reproducible  
- resumable  
- debuggable  
- robust to partial reruns  

### 3.2 SEPARATION OF CONCERNS

```
scripts/         CLI entry points and orchestration only
src/io/          paths, logging, HDF5 helpers, schema
src/geometry/    grids, vessel, coils, boundaries
src/physics/     GS solvers, operators, fields
src/optimization/ PF current fitting
src/viz/         plotting utilities only
```

Scripts contain **NO physics**.  
Physics code **NEVER touches the filesystem**.

This separation is enforced to keep complexity manageable as the project grows.

### 3.3 HDF5 AS A FORMAL API

The structure of `results.h5` is treated as a **FORMAL INTERFACE CONTRACT**.

```
/grid/R
/device/coil_greens/psi_per_amp
/target/boundary
/equilibrium/psi
```

If these paths change:

- schema version must change  
- downstream code must adapt explicitly  

This prevents silent breakage.

---

## 4. REPOSITORY STRUCTURE

```
configs/            YAML input files
scripts/            workflow stages (CLI)
src/                core implementation
docs/               detailed documentation
data/runs/          auto-generated run directories
```

---

## 5. THE RUN CONCEPT AND DATA MODEL

```
data/runs/<RUN_ID>/
  inputs/       archived YAML configs
  figures/      plots and reports
  run.log       full execution log
  results.h5    canonical data store
```

Key ideas:

- inputs/ freezes configuration for reproducibility  
- results.h5 contains **ALL scientific data**  
- run.log contains **ALL logs**  
- rerunning stages updates results.h5, not ad-hoc files  

The HDF5 file also contains:

- /meta/*     metadata (run id, schema version, notes, etc.)  
- /history/*  snapshots of overwritten data for provenance  

---

## 6. WORKFLOW STAGES (SCRIPTS)

### STAGE 00 — INITIALIZE A RUN
Script: `00_init_run.py`

- generate unique run ID  
- archive YAML input configs  
- write metadata and schema version  
- initialize /history  

### STAGE 01 — BUILD THE DEVICE
Script: `01_build_device.py`

- R–Z computational grid  
- vacuum vessel boundary  
- PF coil geometry and current limits  
- coil → ψ Green’s functions  

Defines the **MACHINE CONTRACT**.

### STAGE 02 — DEFINE THE TARGET PLASMA BOUNDARY
Script: `02_target_boundary.py`

- target LCFS boundary  
- boundary flux value  
- global scalar targets (e.g. Ip)  

### STAGE 03 — FIXED-BOUNDARY GS EQUILIBRIUM
Script: `03_solve_fixed_gs.py`

- finite-difference discretization  
- sparse linear solve  
- fixed Dirichlet boundary condition  

Produces `/equilibrium/psi`.

### STAGE 04 — FIT PF COIL CURRENTS
Script: `04_fit_pf_currents.py`

- ridge-regularized least squares  
- optional scaling by current limits  
- enforcement of engineering bounds  
- history snapshots  

Updates `/device/coils/I_pf`.

### STAGE 09 — QUICKLOOK DIAGNOSTICS
Script: `09_quicklook.py`

Reads only `results.h5`  
Writes to `run_dir/figures/`

Workflow driver: `run_workflow.py`

---

## 7. CORE IMPLEMENTATION LAYERS (SRC/)

### src/io/
- run directories  
- logging  
- HDF5 helpers  
- schema validation  

### src/geometry/
- grids  
- coils  
- vessel boundaries  
- plasma boundaries  

### src/physics/
- analytic Green’s functions  
- GS operators  
- fixed-boundary GS solvers  
- field reconstruction  
- derived quantities  

### src/optimization/
- PF current fitting algorithms  

### src/viz/
- plotting utilities only (no physics)  

Each layer is documented in detail in `docs/src/`.

---

## 8. TYPICAL USAGE PATTERNS

```bash
python scripts/run_workflow.py   --device configs/baseline_device.yaml   --target configs/target_equilibrium.yaml   --solver configs/solver.yaml   --to 03
```

Iterate:

```bash
python scripts/run_workflow.py ... --to 03 --overwrite
```

Generate plots:

```bash
python scripts/09_quicklook.py --run-dir data/runs/<RUN_ID>
```

---

## 9. EXTENDING THE WORKFLOW

The architecture is designed to support:

- free-boundary GS solvers  
- nonlinear profile iteration  
- plasma–coil coupling loops  
- stability analysis  
- advanced optimization objectives  

New stages should:

- read from results.h5  
- write new groups/datasets  
- validate schema  
- avoid hidden state  

---

## 10. DOCUMENTATION AND ONBOARDING

This document is the **INSTRUCTION MANUAL**.

Additional documentation lives in `docs/`:

### docs/src/
Pass A / Pass B documentation for each src module  
(design philosophy, physics, numerical algorithms)

### docs/scripts/
One document per script describing:

- inputs and outputs  
- dataflow  
- numerical meaning  
- failure modes  

Recommended onboarding order:

1. This document  
2. docs overview  
3. scripts in execution order  
4. src modules as needed  

---

## 11. CURRENT STATUS AND FUTURE ROADMAP

### CURRENT STATUS

- geometry and targets implemented  
- fixed-boundary equilibria working  
- PF coil fitting implemented  
- diagnostics and plotting in place  

### FUTURE EXPANSION

- free-boundary equilibria  
- stability analysis  
- more realistic plasma physics  
- tighter coupling of equilibrium and optimization  
- richer diagnostics and reporting  

---

## 12. BOTTOM LINE

This repository is not **just a solver**.

It is a **TOKAMAK DESIGN WORKFLOW**:

- explicit  
- reproducible  
- extensible  
- built to grow from machine design  
  all the way to plasma physics  

The current state provides a solid, inspectable base on which much more
sophisticated tokamak design and physics capabilities can be built.
