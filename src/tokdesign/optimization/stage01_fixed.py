"""
tokdesign.optimization.stage01_fixed
====================================

tokdesign.optimization.stage01_fixed
====================================

Stage 01 core: equilibrium-first optimization with fixed-boundary Grad–Shafranov solves.

This module intentionally contains:
  • the stage-level "problem" object
  • a default optimization driver (simple but robust)
  • output writer to the Stage 01 HDF5 schema

It intentionally does NOT contain the physics itself. It expects the following
library functions to exist (you will implement them in subsequent steps):

Required external functions (planned, not yet implemented in your repo):
-----------------------------------------------------------------------
1) Control mapping (from equilibrium_space.yaml)
   - tokdesign.optimization.controls.build_control_mapping(cfg_space) -> ControlMapping
     ControlMapping fields:
       * x_names: list[str]
       * x_bounds_lo: np.ndarray shape (n,)
       * x_bounds_hi: np.ndarray shape (n,)
       * x_init: np.ndarray shape (n,)
       * n_active: int
       * x_to_params(x) -> dict (structured parameters for GS + profiles etc.)
       * params_to_x(params) -> np.ndarray (optional)

2) Grid build (from equilibrium_optimization.yaml numerics/grid)
   - tokdesign.physics.gs.grid.build_grid(cfg_opt, cfg_space) -> Grid
     Grid fields or dict keys:
       * R (1D nR), Z (1D nZ), RR (2D nZ,nR), ZZ (2D nZ,nR), dR, dZ

3) Fixed-boundary GS solve + diagnostics
   - tokdesign.physics.equilibrium.solve_fixed_equilibrium(params, grid, cfg_opt) -> EquilibriumResult
     EquilibriumResult should provide:
       * psi (2D nZ,nR), psi_axis (scalar), psi_lcfs (scalar)
       * axis_R, axis_Z
       * lcfs_R (1D), lcfs_Z (1D)
       * plasma_mask (2D bool/int8)
       * j_phi (2D)
       * fields: BR,BZ,Bphi (2D each)
       * profiles: psi_bar,rho,p,F,q,s,alpha (1D each, same length)
       * diagnostics: gs_iterations (int), residual_norm (float)
       * scalars: dict[str,float] for all scalar metrics needed in trace

4) Objective + constraints evaluation
   - tokdesign.optimization.objectives.evaluate_objective(eq, cfg_opt) -> (total, blocks, terms)
     where blocks, terms are dict[name->float]
   - tokdesign.optimization.constraints.build_constraints(cfg_opt) -> ConstraintSet
     ConstraintSet.evaluate(metric_registry) -> (names, margins, ok)
     where:
       margins: np.ndarray (n_constraints,) float, convention margin>=0 satisfied
       ok: np.ndarray (n_constraints,) bool

This file is written so it becomes fully runnable once those functions exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import math

import numpy as np
import h5py

from tokdesign.io.h5 import (
    h5_ensure_group,
    h5_write_scalar,
    h5_write_array,
    h5_write_strings,
)

from tokdesign.geometry.grid import build_grid
from tokdesign.physics.equilibrium import solve_fixed_equilibrium
from tokdesign.optimization.controls import build_control_mapping
from tokdesign.optimization.objectives import evaluate_objective
from tokdesign.optimization.constraints import build_constraints, ConstraintSet


# =============================================================================
# Failure codes (stable enumeration)
# =============================================================================

FAIL_OK = 0
FAIL_EXCEPTION = 1
FAIL_GS_NOT_CONVERGED = 2
FAIL_GEOMETRY_INVALID = 3
FAIL_NO_LCFS = 4
FAIL_NO_AXIS = 5
FAIL_NAN_INF = 6
FAIL_CONSTRAINT_VIOLATION = 7
FAIL_OTHER_INFEASIBLE = 9

_FAIL_REASON_DEFAULT = {
    FAIL_OK: "ok",
    FAIL_EXCEPTION: "exception",
    FAIL_GS_NOT_CONVERGED: "gs_not_converged",
    FAIL_GEOMETRY_INVALID: "geometry_invalid",
    FAIL_NO_LCFS: "no_lcfs",
    FAIL_NO_AXIS: "no_axis",
    FAIL_NAN_INF: "nan_or_inf",
    FAIL_CONSTRAINT_VIOLATION: "constraint_violation",
    FAIL_OTHER_INFEASIBLE: "infeasible",
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Grid:
    """Lightweight grid container (whatever your grid builder returns)."""
    R: np.ndarray   # (nR,)
    Z: np.ndarray   # (nZ,)
    RR: np.ndarray  # (nZ,nR)
    ZZ: np.ndarray  # (nZ,nR)
    dR: float
    dZ: float


@dataclass
class ControlMapping:
    """
    Minimal interface contract expected from tokdesign.optimization.controls.
    """
    x_names: List[str]
    x_bounds_lo: np.ndarray
    x_bounds_hi: np.ndarray
    x_init: np.ndarray

    def x_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass
class EquilibriumResult:
    """
    Minimal interface contract expected from tokdesign.physics.equilibrium.
    """
    # Fields (best candidate writing)
    psi: np.ndarray
    psi_axis: float
    psi_lcfs: float
    axis_R: float
    axis_Z: float

    lcfs_R: np.ndarray
    lcfs_Z: np.ndarray

    plasma_mask: np.ndarray
    j_phi: np.ndarray

    BR: np.ndarray
    BZ: np.ndarray
    Bphi: np.ndarray

    psi_bar: np.ndarray
    rho: np.ndarray
    p: np.ndarray
    F: np.ndarray
    q: np.ndarray
    s: np.ndarray
    alpha: np.ndarray

    gs_iterations: int
    residual_norm: float

    # Scalars for trace
    scalars: Dict[str, float]


@dataclass
class EvalResult:
    """One evaluation record (one candidate x)."""
    feasible: bool
    fail_code: int
    fail_reason: str

    objective_total: float
    objective_blocks: Dict[str, float]
    objective_terms: Dict[str, float]

    constraint_names: List[str]
    constraint_margins: np.ndarray  # (n_constraints,)
    constraint_ok: np.ndarray       # (n_constraints,)

    metrics: Dict[str, float]       # scalar metrics for trace

    # Optionally include equilibrium artifact (only needed for best tracking)
    eq: Optional[EquilibriumResult] = None


@dataclass
class Stage01Problem:
    """
    Bundle all “static” stage-01 objects:
      • configs
      • control mapping
      • grid
      • evaluation callable
    """
    cfg_space: Dict[str, Any]
    cfg_opt: Dict[str, Any]
    run_context: Dict[str, Any]

    controls: ControlMapping
    grid: Grid

    # Optional hard constraints (parsed once from cfg_opt)
    constraints: Optional[ConstraintSet]

    # "Problem summary" for /stage01_fixed/problem/summary
    summary: str
    active_metrics: List[str]
    active_constraints: List[str]
    active_terms: List[str]


@dataclass
class Stage01Result:
    """Return value of run_optimization()."""
    meta: Dict[str, Any]

    # grid
    grid: Grid

    # problem description
    problem_summary: str
    x_names: List[str]
    x_bounds_lo: np.ndarray
    x_bounds_hi: np.ndarray
    x_init: np.ndarray
    active_metrics: List[str]
    active_constraints: List[str]
    active_terms: List[str]

    # trace arrays
    n_eval: int
    x: np.ndarray                         # (N,n_active)
    feasible: np.ndarray                  # (N,)
    fail_code: np.ndarray                 # (N,)
    fail_reason: List[str]                # (N,) strings

    objective_total: np.ndarray           # (N,)
    objective_blocks: Dict[str, np.ndarray]  # each (N,)
    objective_terms: Dict[str, np.ndarray]   # each (N,)

    constraint_names: List[str]
    constraint_margins: np.ndarray        # (N,n_constraints)
    constraint_ok: np.ndarray             # (N,n_constraints)

    metrics: Dict[str, np.ndarray]        # each (N,)

    # best
    best_eval_index: int
    best_x: np.ndarray
    best_objective_total: float
    best_constraint_margins: np.ndarray
    best_metrics: Dict[str, float]
    best_eq: EquilibriumResult


class MetricRegistry:
    """
    Minimal metric registry for constraints evaluation.

    Prefer eq.scalars; fall back to objectives.get_metric() if needed later.
    """
    def __init__(self, eq):
        self.eq = eq

    def get(self, name: str):
        scalars = getattr(self.eq, "scalars", None)
        if isinstance(scalars, dict) and name in scalars:
            return scalars[name]
        raise KeyError(f"Metric not available: {name}")


# =============================================================================
# Build problem
# =============================================================================

def build_problem(
    cfg_space: Dict[str, Any],
    cfg_opt: Dict[str, Any],
    run_context: Dict[str, Any],
) -> Stage01Problem:
    """
    Build the Stage01Problem:
      - control mapping from equilibrium_space
      - grid from equilibrium_optimization numerics
      - lists of active metrics/constraints/terms (for reporting and schema fields)

    This function should be deterministic.
    """
    # ---- controls ----
    controls = build_control_mapping(cfg_space)

    # ---- grid ----
    grid_obj = build_grid(cfg_opt=cfg_opt, cfg_space=cfg_space)
    grid = _coerce_grid(grid_obj)

    # ---- active lists ----
    active_metrics = _extract_active_metrics(cfg_opt)

    # Parse hard constraints once (matches equilibrium_optimization.yaml)
    constraint_set = build_constraints(cfg_opt, strict=True)
    # Report only enabled constraints if available
    active_constraints = constraint_set.names if constraint_set is not None else _extract_active_constraints(cfg_opt)

    active_terms = _extract_active_terms(cfg_opt)

    # ---- summary ----
    summary = _make_problem_summary(cfg_space, cfg_opt, controls, grid, active_metrics, active_constraints, active_terms)

    return Stage01Problem(
        cfg_space=cfg_space,
        cfg_opt=cfg_opt,
        run_context=run_context,
        controls=controls,
        grid=grid,
        constraints=constraint_set,
        summary=summary,
        active_metrics=active_metrics,
        active_constraints=active_constraints,
        active_terms=active_terms,
    )

# ... rest of file unchanged except:
# - evaluate_candidate() now evaluates constraints via problem.constraints and applies policy
# - write_outputs() now writes required best metrics via _write_required_best_metrics()
# - added _write_required_best_metrics() helper
# - added radial coordinate design comment in _write_best_equilibrium()



def _coerce_grid(grid_obj: Any) -> Grid:
    """
    Accept either a dict-like grid or an object with attributes.
    Coerce into our Grid dataclass.
    """
    if isinstance(grid_obj, dict):
        return Grid(
            R=np.asarray(grid_obj["R"]),
            Z=np.asarray(grid_obj["Z"]),
            RR=np.asarray(grid_obj["RR"]),
            ZZ=np.asarray(grid_obj["ZZ"]),
            dR=float(grid_obj["dR"]),
            dZ=float(grid_obj["dZ"]),
        )
    # attribute style
    return Grid(
        R=np.asarray(getattr(grid_obj, "R")),
        Z=np.asarray(getattr(grid_obj, "Z")),
        RR=np.asarray(getattr(grid_obj, "RR")),
        ZZ=np.asarray(getattr(grid_obj, "ZZ")),
        dR=float(getattr(grid_obj, "dR")),
        dZ=float(getattr(grid_obj, "dZ")),
    )


def _extract_active_metrics(cfg_opt: Dict[str, Any]) -> List[str]:
    """
    Best-effort: collect metric names from cfg_opt["metrics"].
    You can refine once your optimization YAML stabilizes.
    """
    m = cfg_opt.get("metrics", {})
    if isinstance(m, dict):
        return sorted(list(m.keys()))
    return []


def _extract_active_constraints(cfg_opt: Dict[str, Any]) -> List[str]:
    """
    Best-effort: constraints list is often cfg_opt["constraints"]["list"] (your YAML).
    """
    c = cfg_opt.get("constraints", {})
    if isinstance(c, dict):
        lst = c.get("list", [])
        if isinstance(lst, list):
            out = []
            for item in lst:
                if isinstance(item, dict) and "name" in item:
                    out.append(str(item["name"]))
            return out
    return []


def _extract_active_terms(cfg_opt: Dict[str, Any]) -> List[str]:
    """
    Best-effort: objective terms in cfg_opt["objective"]["blocks"][...]["terms"].
    """
    obj = cfg_opt.get("objective", {})
    out: List[str] = []
    if isinstance(obj, dict):
        blocks = obj.get("blocks", [])
        if isinstance(blocks, list):
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                terms = b.get("terms", [])
                if isinstance(terms, list):
                    for t in terms:
                        if isinstance(t, dict) and "name" in t:
                            out.append(str(t["name"]))
    return out


def _make_problem_summary(
    cfg_space: Dict[str, Any],
    cfg_opt: Dict[str, Any],
    controls: ControlMapping,
    grid: Grid,
    active_metrics: List[str],
    active_constraints: List[str],
    active_terms: List[str],
) -> str:
    """
    Human-readable 1–2 paragraph summary for /stage01_fixed/problem/summary.
    Keep it short and stable.
    """
    nR = int(grid.R.shape[0])
    nZ = int(grid.Z.shape[0])

    lines = []
    lines.append("Stage 01: fixed-boundary equilibrium optimization (Grad–Shafranov).")
    lines.append(f"Active controls: {len(controls.x_names)}")
    lines.append(f"Grid: nR={nR}, nZ={nZ}, dR={grid.dR:.4g}, dZ={grid.dZ:.4g}")
    if active_metrics:
        lines.append(f"Metrics: {len(active_metrics)} (e.g. {', '.join(active_metrics[:6])}{'...' if len(active_metrics)>6 else ''})")
    if active_constraints:
        lines.append(f"Constraints: {len(active_constraints)} (e.g. {', '.join(active_constraints[:6])}{'...' if len(active_constraints)>6 else ''})")
    if active_terms:
        lines.append(f"Objective terms: {len(active_terms)} (e.g. {', '.join(active_terms[:6])}{'...' if len(active_terms)>6 else ''})")
    return "\n".join(lines)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_candidate(problem: Stage01Problem, x: np.ndarray) -> EvalResult:
    """
    Evaluate a single candidate control vector x.

    Pipeline:
      1) x -> structured params (controls mapping)
      2) solve fixed-boundary equilibrium (GS / stub)
      3) compute constraints and objective
      4) assemble EvalResult (including scalar metrics for trace)

    Notes
    -----
    • Hard failures (controls mapping, solver crash, NaN/Inf equilibrium) return FAIL_* and
      objective_total = +inf.
    • Constraints are handled via ConstraintSet (problem.constraints):
        - policy == "reject": violated => infeasible, objective_total = +inf (cheap gate)
        - policy == "penalty": violated => still evaluate objective, then add quadratic penalty
          from negative margins (smooth guidance toward feasibility)
    • We return eq only for feasible candidates by default (memory). The "best" equilibrium can
      always be recomputed from best_x at the end of optimization if needed.
    """
    x = np.asarray(x, dtype=float).reshape(-1)

    # ---------------------------
    # 1) Map x -> params
    # ---------------------------
    try:
        params = problem.controls.x_to_params(x)
    except Exception as e:
        return _fail(problem, x, FAIL_EXCEPTION, f"controls_mapping_exception: {e}")

    # ---------------------------
    # 2) Solve equilibrium
    # ---------------------------
    try:
        eq = _solve_equilibrium(problem, params)
    except NotImplementedError:
        # If you intentionally raise for missing implementation, do not swallow it.
        raise
    except Exception as e:
        return _fail(problem, x, FAIL_EXCEPTION, f"solve_exception: {e}")

    # Basic sanity (avoid NaNs in objective/constraints)
    if not _finite_equilibrium(eq):
        return _fail(problem, x, FAIL_NAN_INF, "nan_or_inf_in_equilibrium")

    # ---------------------------
    # 3) Constraints (new system)
    # ---------------------------
    constraint_names: list[str] = []
    c_margins = np.zeros((0,), dtype=float)
    c_ok = np.zeros((0,), dtype=bool)

    policy = "reject"
    if problem.constraints is not None:
        policy = str(problem.constraints.on_violation).strip().lower()
        try:
            constraint_names = problem.constraints.names
            mr = MetricRegistry(eq)
            ce = problem.constraints.evaluate(mr)
            c_margins = np.asarray(ce.margins, dtype=float)
            c_ok = np.asarray(ce.ok, dtype=bool)
        except Exception as e:
            # If constraints evaluation fails, this is a hard failure (cannot trust metrics).
            return _fail(problem, x, FAIL_EXCEPTION, f"constraints_exception: {e}")

    feasible = bool(np.all(c_ok)) if c_ok.size else True
    
        # ---- diagnostics: print violated constraints (optional)
    if (c_ok.size > 0) and (not feasible):
        violated = [(n, float(m)) for n, m, ok in zip(constraint_names, c_margins, c_ok) if not bool(ok)]
        msg = ", ".join(f"{n} (margin={m:+.3e})" for n, m in violated)
        print(f"[constraints] violated: {msg}")

    # If constraints are hard-reject and violated, skip expensive objective evaluation
    if (problem.constraints is not None) and (policy == "reject") and (not feasible):
        return EvalResult(
            feasible=False,
            fail_code=FAIL_CONSTRAINT_VIOLATION,
            fail_reason=_FAIL_REASON_DEFAULT[FAIL_CONSTRAINT_VIOLATION],
            objective_total=float("inf"),
            objective_blocks={},
            objective_terms={},
            constraint_names=constraint_names,
            constraint_margins=c_margins,
            constraint_ok=c_ok,
            metrics=_extract_trace_metrics(eq),
            eq=None,
        )

    # ---------------------------
    # 4) Objective
    # ---------------------------
    try:
        # Use the new objective evaluator (YAML-aligned).
        # We pass context so meta-metrics like "distance_to_bounds" work.
        f_total, f_blocks, f_terms = evaluate_objective(
            eq=eq,
            cfg_opt=problem.cfg_opt,
            context={
                "x": x,
                "x_bounds_lo": np.asarray(problem.controls.x_bounds_lo, dtype=float).reshape(-1),
                "x_bounds_hi": np.asarray(problem.controls.x_bounds_hi, dtype=float).reshape(-1),
                "x_names": list(problem.controls.x_names),
            },
        )
    except Exception as e:
        return _fail(problem, x, FAIL_EXCEPTION, f"objective_exception: {e}")

    f_total = float(f_total)
    # Penalty mode: add smooth penalty for violated constraints
    if (problem.constraints is not None) and (policy == "penalty") and (not feasible):
        # Simple quadratic penalty on negative margins:
        #   v_i = max(0, -margin_i)
        #   penalty = sum(v_i^2)
        # This is dimensioned in "margin units"; later you can add per-constraint scales/weights.
        v = np.maximum(-c_margins, 0.0)
        penalty = float(np.sum(v * v))
        f_total = float(f_total + penalty)

    # ---------------------------
    # Assemble EvalResult
    # ---------------------------
    if not feasible:
        return EvalResult(
            feasible=False,
            fail_code=FAIL_CONSTRAINT_VIOLATION,
            fail_reason=_FAIL_REASON_DEFAULT[FAIL_CONSTRAINT_VIOLATION],
            objective_total=float(f_total),
            objective_blocks=f_blocks,
            objective_terms=f_terms,
            constraint_names=constraint_names,
            constraint_margins=c_margins,
            constraint_ok=c_ok,
            metrics=_extract_trace_metrics(eq),
            eq=None,  # keep memory light; recompute best equilibrium later if needed
        )

    # Feasible candidate
    return EvalResult(
        feasible=True,
        fail_code=FAIL_OK,
        fail_reason=_FAIL_REASON_DEFAULT[FAIL_OK],
        objective_total=float(f_total),
        objective_blocks=f_blocks,
        objective_terms=f_terms,
        constraint_names=constraint_names,
        constraint_margins=c_margins,
        constraint_ok=c_ok,
        metrics=_extract_trace_metrics(eq),
        eq=eq,  # optionally keep for feasible candidates; best can also be recomputed later
    )


def _solve_equilibrium(problem: Stage01Problem, params: Dict[str, Any]) -> EquilibriumResult:
    """
    Call the physics layer to solve the fixed-boundary equilibrium.
    """
    eq_obj = solve_fixed_equilibrium(params=params, grid=problem.grid, cfg_opt=problem.cfg_opt)
    return _coerce_equilibrium(eq_obj)


def _coerce_equilibrium(eq_obj: Any) -> EquilibriumResult:
    """
    Accept either dict-like or attribute-like equilibrium result and coerce.
    """
    if isinstance(eq_obj, dict):
        fields = eq_obj.get("fields", {})
        prof = eq_obj.get("profiles", {})
        diag = eq_obj.get("diagnostics", {})
        return EquilibriumResult(
            psi=np.asarray(eq_obj["psi"]),
            psi_axis=float(eq_obj["psi_axis"]),
            psi_lcfs=float(eq_obj["psi_lcfs"]),
            axis_R=float(eq_obj["axis_R"]),
            axis_Z=float(eq_obj["axis_Z"]),
            lcfs_R=np.asarray(eq_obj["lcfs_R"]),
            lcfs_Z=np.asarray(eq_obj["lcfs_Z"]),
            plasma_mask=np.asarray(eq_obj["plasma_mask"]),
            j_phi=np.asarray(eq_obj["j_phi"]),
            BR=np.asarray(fields["BR"]),
            BZ=np.asarray(fields["BZ"]),
            Bphi=np.asarray(fields["Bphi"]),
            psi_bar=np.asarray(prof["psi_bar"]),
            rho=np.asarray(prof["rho"]),
            p=np.asarray(prof["p"]),
            F=np.asarray(prof["F"]),
            q=np.asarray(prof["q"]),
            s=np.asarray(prof["s"]),
            alpha=np.asarray(prof["alpha"]),
            gs_iterations=int(diag.get("gs_iterations", 0)),
            residual_norm=float(diag.get("residual_norm", np.nan)),
            scalars=dict(eq_obj.get("scalars", {})),
        )

    # attribute style
    return EquilibriumResult(
        psi=np.asarray(getattr(eq_obj, "psi")),
        psi_axis=float(getattr(eq_obj, "psi_axis")),
        psi_lcfs=float(getattr(eq_obj, "psi_lcfs")),
        axis_R=float(getattr(eq_obj, "axis_R")),
        axis_Z=float(getattr(eq_obj, "axis_Z")),
        lcfs_R=np.asarray(getattr(eq_obj, "lcfs_R")),
        lcfs_Z=np.asarray(getattr(eq_obj, "lcfs_Z")),
        plasma_mask=np.asarray(getattr(eq_obj, "plasma_mask")),
        j_phi=np.asarray(getattr(eq_obj, "j_phi")),
        BR=np.asarray(getattr(eq_obj, "BR")),
        BZ=np.asarray(getattr(eq_obj, "BZ")),
        Bphi=np.asarray(getattr(eq_obj, "Bphi")),
        psi_bar=np.asarray(getattr(eq_obj, "psi_bar")),
        rho=np.asarray(getattr(eq_obj, "rho")),
        p=np.asarray(getattr(eq_obj, "p")),
        F=np.asarray(getattr(eq_obj, "F")),
        q=np.asarray(getattr(eq_obj, "q")),
        s=np.asarray(getattr(eq_obj, "s")),
        alpha=np.asarray(getattr(eq_obj, "alpha")),
        gs_iterations=int(getattr(eq_obj, "gs_iterations", 0)),
        residual_norm=float(getattr(eq_obj, "residual_norm", np.nan)),
        scalars=dict(getattr(eq_obj, "scalars", {})),
    )


def _evaluate_objective(eq: EquilibriumResult, cfg_opt: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Call objective evaluation.
    """

    f_total, f_blocks, f_terms = evaluate_objective(eq=eq, cfg_opt=cfg_opt)
    # force plain floats
    f_total = float(f_total)
    f_blocks = {str(k): float(v) for k, v in dict(f_blocks).items()}
    f_terms = {str(k): float(v) for k, v in dict(f_terms).items()}
    return f_total, f_blocks, f_terms


def _evaluate_constraints(eq: EquilibriumResult, cfg_opt: Dict[str, Any]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Call constraint evaluation.
    """

    names, margins, ok = evaluate_constraints(eq=eq, cfg_opt=cfg_opt)
    names = [str(n) for n in names]
    margins = np.asarray(margins, dtype=float).reshape(-1)
    ok = np.asarray(ok, dtype=bool).reshape(-1)
    return names, margins, ok


def _extract_trace_metrics(eq: EquilibriumResult) -> Dict[str, float]:
    """
    Stage 01 trace requires many scalars. We take them from eq.scalars.

    This keeps stage01_fixed independent of the specific metric implementations.
    You will ensure solve_fixed_equilibrium() populates eq.scalars with the
    required keys.

    Required keys are listed in schema.py and your stage01 schema list.
    """
    if not isinstance(eq.scalars, dict):
        return {}

    # Convert to float where possible (int is fine too, but float is safe)
    out: Dict[str, float] = {}
    for k, v in eq.scalars.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # If a scalar is missing or non-numeric, skip it; schema writer will fill NaNs.
            continue
    return out


def _finite_equilibrium(eq: EquilibriumResult) -> bool:
    """
    Basic numerical sanity checks to avoid poisoning the optimizer trace with NaNs.
    """
    arrs = [
        eq.psi, eq.plasma_mask, eq.j_phi,
        eq.BR, eq.BZ, eq.Bphi,
        eq.psi_bar, eq.rho, eq.p, eq.F, eq.q, eq.s, eq.alpha
    ]
    for a in arrs:
        aa = np.asarray(a)
        if np.any(~np.isfinite(aa)):
            return False
    scalars = [eq.psi_axis, eq.psi_lcfs, eq.axis_R, eq.axis_Z, eq.residual_norm]
    for s in scalars:
        if not np.isfinite(float(s)):
            return False
    return True


def _fail(problem: Stage01Problem, x: np.ndarray, code: int, reason: str) -> EvalResult:
    """
    Helper to create a failure EvalResult with safe defaults.
    """
    # Constraint placeholders
    c_names = problem.active_constraints[:] if problem.active_constraints else []
    n_c = len(c_names)
    c_margins = np.full((n_c,), np.nan, dtype=float)
    c_ok = np.zeros((n_c,), dtype=bool)

    return EvalResult(
        feasible=False,
        fail_code=int(code),
        fail_reason=str(reason) if reason else _FAIL_REASON_DEFAULT.get(code, "fail"),
        objective_total=float("inf"),
        objective_blocks={},
        objective_terms={},
        constraint_names=c_names,
        constraint_margins=c_margins,
        constraint_ok=c_ok,
        metrics={},
        eq=None,
    )


# =============================================================================
# Optimization driver (default: bounded random search + local perturbations)
# =============================================================================

def run_optimization(
    problem: Stage01Problem,
    optimizer: Optional[str] = None,
    max_evals: Optional[int] = None,
    seed: Optional[int] = None,
) -> Stage01Result:
    """
    Run Stage 01 optimization.

    Default behavior is deliberately simple and robust:
      • random sampling inside bounds (global exploration)
      • plus local Gaussian perturbations around the current best (exploitation)

    This is not meant to be the final optimizer; it is meant to be:
      • deterministic with a seed
      • stable for development
      • produces full trace according to schema

    Later you can replace this with CMA-ES, Bayesian opt, etc. without changing
    the HDF5 schema or stage orchestrator.
    """
    opt_name = (optimizer or "").strip().lower() or "random_local"

    allow_infeasible_best = (
        problem.constraints is not None
        and getattr(problem.constraints, "on_violation", "").lower() == "penalty"
    )

    # Pull defaults from cfg_opt if not overridden
    cfg_run = problem.cfg_opt.get("optimizer", {}) if isinstance(problem.cfg_opt.get("optimizer", {}), dict) else {}
    if max_evals is None:
        max_evals = int(cfg_run.get("max_evals", 4))
    if seed is None:
        seed_val = cfg_run.get("seed", 0)
        seed = int(seed_val) if isinstance(seed_val, (int, float)) else 0

    rng = np.random.default_rng(seed)

    controls = problem.controls
    n = len(controls.x_names)
    lo = np.asarray(controls.x_bounds_lo, dtype=float).reshape(n)
    hi = np.asarray(controls.x_bounds_hi, dtype=float).reshape(n)
    x0 = np.asarray(controls.x_init, dtype=float).reshape(n)

    # Trace buffers (append then stack once)
    xs: List[np.ndarray] = []
    feasible: List[int] = []
    fail_code: List[int] = []
    fail_reason: List[str] = []

    obj_total: List[float] = []
    obj_blocks_hist: Dict[str, List[float]] = {}
    obj_terms_hist: Dict[str, List[float]] = {}

    constraint_names: List[str] = []
    c_margins_list: List[np.ndarray] = []
    c_ok_list: List[np.ndarray] = []

    metric_hist: Dict[str, List[float]] = {}

    # Best tracking
    best_idx = -1
    best_f = float("inf")
    best_eq: Optional[EquilibriumResult] = None
    best_metrics: Dict[str, float] = {}
    best_c_margins: Optional[np.ndarray] = None
    best_x: Optional[np.ndarray] = None

    # Heuristic schedule: half global, half local
    n_global = max_evals // 2
    n_local = max_evals - n_global

    # Local step size: fraction of bounds width
    widths = np.maximum(hi - lo, 1e-12)
    sigma0 = 0.15 * widths

    # A helper to clamp to bounds
    def clip_to_bounds(x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(x, lo), hi)

    # Evaluate first point at x_init (always do this)
    candidates: List[np.ndarray] = [clip_to_bounds(x0.copy())]

    # Global random samples
    for _ in range(n_global):
        u = rng.random(n)
        candidates.append(lo + u * (hi - lo))

    # Local perturbations around best-so-far (start around x0; updated as best improves)
    # We generate them now but they will be evaluated sequentially, and we will
    # update the center as best changes.
    # For local, we will sample on-the-fly below.

    # ---- evaluation loop ----
    eval_k = 0
    # Evaluate x_init + global samples first
    for x in candidates:
        er = evaluate_candidate(problem, x)
        _append_eval(
            er, x, xs, feasible, fail_code, fail_reason,
            obj_total, obj_blocks_hist, obj_terms_hist,
            constraint_names, c_margins_list, c_ok_list,
            metric_hist
        )
        #if eval_k%12==0: print(f"er (k={eval_k}) = {er}")

        print(f"progress:{eval_k} out of {max_evals} --> ~ {(100*eval_k/max_evals):.3g}%")
        #print(f"for x({eval_k}) i candidate(global): er.eq={er.eq is not None},  and (er.objective_total < best_f)={er.objective_total < best_f} and (er.feasible or allow_infeasible_best)={(er.feasible or allow_infeasible_best)}")
        #print(f"er({eval_k})={er}")
        if (er.eq is not None) and (er.objective_total < best_f) and (er.feasible or allow_infeasible_best):
            best_f = float(er.objective_total)
            best_idx = eval_k
            best_eq = er.eq
            best_metrics = dict(er.metrics)
            best_c_margins = np.asarray(er.constraint_margins, dtype=float).copy()
            best_x = np.asarray(x, dtype=float).copy()

        eval_k += 1

    # Now local phase: sequentially propose perturbations around current best
    # If no feasible best exists yet, perturb around x0.
    for i in range(n_local):
        center = best_x if best_x is not None else x0
        # anneal sigma a bit with i
        t = i / max(1, n_local - 1)
        sigma = sigma0 * (0.8 * (1.0 - t) + 0.2)  # decays to 0.2*sigma0

        x = center + rng.normal(0.0, 1.0, size=n) * sigma
        x = clip_to_bounds(x)

        er = evaluate_candidate(problem, x)
        _append_eval(
            er, x, xs, feasible, fail_code, fail_reason,
            obj_total, obj_blocks_hist, obj_terms_hist,
            constraint_names, c_margins_list, c_ok_list,
            metric_hist
        )

        print(f"progress:{eval_k} out of {max_evals} --> ~ {(100*eval_k/max_evals):.3g}%")
        #print(f"for x({eval_k}) i candidate(local): er.eq={er.eq is not None},  and (er.objective_total < best_f)={er.objective_total < best_f} and (er.feasible or allow_infeasible_best)={(er.feasible or allow_infeasible_best)}")
        if (er.eq is not None) and (er.objective_total < best_f) and (er.feasible or allow_infeasible_best):
            best_f = float(er.objective_total)
            best_idx = eval_k
            best_eq = er.eq
            best_metrics = dict(er.metrics)
            best_c_margins = np.asarray(er.constraint_margins, dtype=float).copy()
            best_x = np.asarray(x, dtype=float).copy()

        eval_k += 1
        #print(f"for x({eval_k}) i candidate")

    if best_x is None:
        # Nothing evaluated (should not happen unless max_evals=0)
        raise RuntimeError("Stage 01 produced no evaluations (max_evals may be  or no equilibrium passed constraints).")

    if best_eq is None or best_c_margins is None:
        # No feasible candidate found (or best never stored). Choose lowest objective_total anyway.
        best_idx = int(np.nanargmin(np.asarray(obj_total, dtype=float)))
        best_x = xs[best_idx].copy()

        # Recompute equilibrium for best_x (so /best/equilibrium can be written)
        best_params = controls.x_to_params(best_x)
        best_eq = solve_fixed_equilibrium(params=best_params, grid=problem.grid, cfg_opt=problem.cfg_opt)

        # Recompute constraints for best (for /best/constraints_margins)
        if problem.constraints is not None:
            mr = MetricRegistry(best_eq)
            ce = problem.constraints.evaluate(mr)
            best_c_margins = np.asarray(ce.margins, dtype=float)
        else:
            best_c_margins = np.zeros((0,), dtype=float)

        # Best metrics: use eq.scalars (fallback to NaN later in writer)
        scalars = getattr(best_eq, "scalars", {})
        best_metrics = dict(scalars) if isinstance(scalars, dict) else {}

        # Best objective value should match trace at best_idx
        best_f = float(obj_total[best_idx])


    # Stack trace arrays
    X = np.vstack(xs).astype(float)
    feasible_arr = np.asarray(feasible, dtype=np.int8)
    fail_code_arr = np.asarray(fail_code, dtype=int)
    obj_total_arr = np.asarray(obj_total, dtype=float)

    # Blocks/terms: convert lists -> arrays, fill missing with NaNs
    block_arrays = {k: np.asarray(v, dtype=float) for k, v in obj_blocks_hist.items()}
    term_arrays = {k: np.asarray(v, dtype=float) for k, v in obj_terms_hist.items()}

    # Constraints: ensure consistent ordering
    if constraint_names:
        CM = np.vstack(c_margins_list).astype(float)
        COK = np.vstack(c_ok_list).astype(np.int8)
    else:
        CM = np.zeros((X.shape[0], 0), dtype=float)
        COK = np.zeros((X.shape[0], 0), dtype=np.int8)

    # Metrics arrays: fill missing keys with NaNs
    metric_arrays = {k: np.asarray(v, dtype=float) for k, v in metric_hist.items()}

    meta = {
        "stage": "stage01_fixed",
        "optimizer": opt_name,
        "max_evals": int(max_evals),
        "seed": int(seed),
    }
    # add run_context into meta (non-essential)
    for k, v in (problem.run_context or {}).items():
        if k not in meta:
            meta[k] = v

    return Stage01Result(
        meta=meta,
        grid=problem.grid,
        problem_summary=problem.summary,
        x_names=controls.x_names,
        x_bounds_lo=lo,
        x_bounds_hi=hi,
        x_init=x0,
        active_metrics=problem.active_metrics,
        active_constraints=problem.active_constraints,
        active_terms=problem.active_terms,
        n_eval=int(X.shape[0]),
        x=X,
        feasible=feasible_arr,
        fail_code=fail_code_arr,
        fail_reason=fail_reason,
        objective_total=obj_total_arr,
        objective_blocks=block_arrays,
        objective_terms=term_arrays,
        constraint_names=constraint_names,
        constraint_margins=CM,
        constraint_ok=COK,
        metrics=metric_arrays,
        best_eval_index=int(best_idx),
        best_x=np.asarray(best_x, dtype=float),
        best_objective_total=float(best_f),
        best_constraint_margins=np.asarray(best_c_margins, dtype=float),
        best_metrics=best_metrics,
        best_eq=best_eq,
    )


def _append_eval(
    er: EvalResult,
    x: np.ndarray,
    xs: List[np.ndarray],
    feasible: List[int],
    fail_code: List[int],
    fail_reason: List[str],
    obj_total: List[float],
    obj_blocks_hist: Dict[str, List[float]],
    obj_terms_hist: Dict[str, List[float]],
    constraint_names: List[str],
    c_margins_list: List[np.ndarray],
    c_ok_list: List[np.ndarray],
    metric_hist: Dict[str, List[float]],
) -> None:
    """
    Append an evaluation record into trace buffers, keeping arrays aligned.

    This function ensures:
      • all block/term arrays have length N (filled with NaN when missing)
      • constraints are aligned to a stable ordering (names locked on first eval that returns names)
      • metrics arrays have length N (filled with NaN when missing)
    """
    xs.append(np.asarray(x, dtype=float).copy())
    feasible.append(1 if er.feasible else 0)
    fail_code.append(int(er.fail_code))
    fail_reason.append(str(er.fail_reason))

    obj_total.append(float(er.objective_total))

    # Objective blocks: fill any previously seen blocks with NaN if missing this eval
    _append_named_scalars(obj_blocks_hist, er.objective_blocks)

    # Objective terms
    _append_named_scalars(obj_terms_hist, er.objective_terms)

    # Constraints: lock names on first eval that provides them
    if not constraint_names and er.constraint_names:
        constraint_names[:] = list(er.constraint_names)

    if constraint_names:
        # align margins/ok to locked ordering (missing -> NaN/False)
        name_to_idx = {n: i for i, n in enumerate(er.constraint_names)}
        margins = np.full((len(constraint_names),), np.nan, dtype=float)
        ok = np.zeros((len(constraint_names),), dtype=np.int8)
        for i, n in enumerate(constraint_names):
            j = name_to_idx.get(n, None)
            if j is None:
                continue
            margins[i] = float(er.constraint_margins[j])
            ok[i] = 1 if bool(er.constraint_ok[j]) else 0
        c_margins_list.append(margins)
        c_ok_list.append(ok)
    else:
        # No constraints defined
        c_margins_list.append(np.zeros((0,), dtype=float))
        c_ok_list.append(np.zeros((0,), dtype=np.int8))

    # Metrics: union keys over time, fill NaN for missing at each eval
    _append_named_scalars(metric_hist, er.metrics)


def _append_named_scalars(store: Dict[str, List[float]], values: Dict[str, float]) -> None:
    """
    Maintain aligned histories for a dict of scalar time series.

    store maps name -> list of length N so far.
    values gives name -> scalar for current eval.
    """
    # current length N before adding this eval
    if store:
        N = len(next(iter(store.values())))
    else:
        N = 0

    # ensure all existing keys get a placeholder if missing
    for k in store.keys():
        store[k].append(float(values[k]) if k in values else float("nan"))

    # any new keys: backfill N previous NaNs, then add current value
    for k, v in values.items():
        kk = str(k)
        if kk not in store:
            store[kk] = [float("nan")] * N + [float(v)]



