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


# =============================================================================
# Output writer (matches your schema)
# =============================================================================

def write_outputs(h5: h5py.File, result: Stage01Result) -> None:
    """
    Write Stage 01 outputs to HDF5 under /stage01_fixed following your schema.

    Convention:
      • margins >= 0 is satisfied
      • constraint_ok is stored as int8 (0/1)
      • feasible is stored as int8 (0/1)
      • fail_reason is stored as short strings
      • missing metrics are filled with NaN arrays (but you should aim to provide them)
    """
    # Root
    h5_ensure_group(h5, "/stage01_fixed")

    # ---------------- Grid
    h5_ensure_group(h5, "/stage01_fixed/grid")
    h5_write_array(h5, "/stage01_fixed/grid/R", result.grid.R)
    h5_write_array(h5, "/stage01_fixed/grid/Z", result.grid.Z)
    h5_write_array(h5, "/stage01_fixed/grid/RR", result.grid.RR)
    h5_write_array(h5, "/stage01_fixed/grid/ZZ", result.grid.ZZ)
    h5_write_scalar(h5, "/stage01_fixed/grid/dR", float(result.grid.dR))
    h5_write_scalar(h5, "/stage01_fixed/grid/dZ", float(result.grid.dZ))

    # ---------------- Problem overview
    h5_ensure_group(h5, "/stage01_fixed/problem")
    h5_write_scalar(h5, "/stage01_fixed/problem/summary", str(result.problem_summary))

    # active_controls: simplest stable representation is a group with names/bounds/init arrays
    h5_ensure_group(h5, "/stage01_fixed/problem/active_controls")
    h5_write_strings(h5, "/stage01_fixed/problem/active_controls/names", result.x_names)
    h5_write_array(h5, "/stage01_fixed/problem/active_controls/bounds_lo", result.x_bounds_lo)
    h5_write_array(h5, "/stage01_fixed/problem/active_controls/bounds_hi", result.x_bounds_hi)
    h5_write_array(h5, "/stage01_fixed/problem/active_controls/x_init", result.x_init)

    h5_write_strings(h5, "/stage01_fixed/problem/active_metrics", result.active_metrics)
    h5_write_strings(h5, "/stage01_fixed/problem/active_constraints", result.active_constraints)
    h5_write_strings(h5, "/stage01_fixed/problem/active_terms", result.active_terms)

    # ---------------- Trace
    h5_ensure_group(h5, "/stage01_fixed/trace")
    h5_write_scalar(h5, "/stage01_fixed/trace/n_eval", int(result.n_eval))
    h5_write_array(h5, "/stage01_fixed/trace/x", result.x)
    h5_write_strings(h5, "/stage01_fixed/trace/x_names", result.x_names)
    h5_write_array(h5, "/stage01_fixed/trace/x_bounds_lo", result.x_bounds_lo)
    h5_write_array(h5, "/stage01_fixed/trace/x_bounds_hi", result.x_bounds_hi)
    h5_write_array(h5, "/stage01_fixed/trace/x_init", result.x_init)

    h5_write_array(h5, "/stage01_fixed/trace/feasible", result.feasible.astype(np.int8))
    h5_write_array(h5, "/stage01_fixed/trace/fail_code", result.fail_code.astype(int))
    h5_write_strings(h5, "/stage01_fixed/trace/fail_reason", result.fail_reason)

    h5_write_array(h5, "/stage01_fixed/trace/objective_total", result.objective_total.astype(float))

    # objective blocks
    h5_ensure_group(h5, "/stage01_fixed/trace/objective_blocks")
    for name, arr in sorted(result.objective_blocks.items()):
        h5_write_array(h5, f"/stage01_fixed/trace/objective_blocks/{name}", np.asarray(arr, dtype=float))

    # objective terms
    h5_ensure_group(h5, "/stage01_fixed/trace/objective_terms")
    for name, arr in sorted(result.objective_terms.items()):
        h5_write_array(h5, f"/stage01_fixed/trace/objective_terms/{name}", np.asarray(arr, dtype=float))

    # constraints
    h5_ensure_group(h5, "/stage01_fixed/trace/constraints")
    h5_write_strings(h5, "/stage01_fixed/trace/constraints/names", result.constraint_names)
    h5_write_array(h5, "/stage01_fixed/trace/constraints/margins", result.constraint_margins.astype(float))
    h5_write_array(h5, "/stage01_fixed/trace/constraints/ok", result.constraint_ok.astype(np.int8))

    # metrics scalars
    h5_ensure_group(h5, "/stage01_fixed/trace/metrics")
    _write_required_trace_metrics(h5, result)

    # ---------------- Best candidate
    h5_ensure_group(h5, "/stage01_fixed/best")
    h5_write_scalar(h5, "/stage01_fixed/best/eval_index", int(result.best_eval_index))
    h5_write_array(h5, "/stage01_fixed/best/x", np.asarray(result.best_x, dtype=float))
    h5_write_scalar(h5, "/stage01_fixed/best/objective_total", float(result.best_objective_total))

    h5_write_array(
        h5,
        "/stage01_fixed/best/constraints_margins",
        np.asarray(result.best_constraint_margins, dtype=float),
    )

    # best metrics group (same names as trace scalars; only write those we have)
    h5_ensure_group(h5, "/stage01_fixed/best/metrics")
    for k, v in sorted(result.best_metrics.items()):
        h5_write_scalar(h5, f"/stage01_fixed/best/metrics/{k}", float(v))

    # equilibrium fields for best
    _write_best_equilibrium(h5, result.best_eq)

    # meta (optional but useful)
    h5_ensure_group(h5, "/stage01_fixed/meta")
    for k, v in sorted(result.meta.items()):
        # store all meta fields as strings unless clearly numeric
        if isinstance(v, (int, np.integer)):
            h5_write_scalar(h5, f"/stage01_fixed/meta/{k}", int(v))
        elif isinstance(v, (float, np.floating)):
            h5_write_scalar(h5, f"/stage01_fixed/meta/{k}", float(v))
        else:
            h5_write_scalar(h5, f"/stage01_fixed/meta/{k}", str(v))


def _write_required_trace_metrics(h5: h5py.File, result: Stage01Result) -> None:
    """
    Your schema requires a large set of scalar metrics in /stage01_fixed/trace/metrics/*.

    This helper writes:
      - if present in result.metrics, use it
      - else create NaN arrays of length N (so schema still validates)

    IMPORTANT:
      The actual computation of these scalars should happen in your physics/metrics
      layer and be placed into EquilibriumResult.scalars for each eval.
    """
    required = [
        "I_t","B0","volume","poloidal_flux","stored_energy","aspect_ratio",
        "beta","beta_p","beta_N","li","kappa","delta","shafranov_shift",
        "n_vol_avg", "n_line_avg_midplane", "n_vol_avg_1e20", "n_line_avg_midplane_1e20",
        "n_greenwald_1e20", "greenwald_fraction",
        "Te_max","Te_min","Te_p05","Te_p50","Te_p95",
        "q0","q95","q_min","rho_qmin","low_q_volume_fraction","q_monotonicity_violation",
        "q_rational_proximity","q_smoothness",
        "s_edge_mean","s_edge_min","s_min","s_max","negative_shear_extent","shear_smoothness",
        "alpha_edge_mean","alpha_edge_p95","alpha_edge_integral","s_alpha_margin_min",
        "s_alpha_negative_margin_integral",
        "p_peaking_factor","dpdrho_max","edge_pressure_gradient_integral",
        "j_peaking_factor","current_centroid_shift",
        "fusion_power_MW", "alpha_power_MW", "Q_est",
    ]
    N = int(result.n_eval)
    for name in required:
        if name in result.metrics:
            arr = np.asarray(result.metrics[name], dtype=float).reshape(N)
        else:
            arr = np.full((N,), np.nan, dtype=float)
        h5_write_array(h5, f"/stage01_fixed/trace/metrics/{name}", arr)


def _write_best_equilibrium(h5: h5py.File, eq: EquilibriumResult) -> None:
    """
    Write the heavy equilibrium artifacts for the best candidate:
      - psi, axis, lcfs, mask, current, fields
      - profiles psi_bar/rho/p/F/q/s/alpha
      - diagnostics gs_iterations/residual_norm
    """
    h5_ensure_group(h5, "/stage01_fixed/best/equilibrium")

    h5_write_array(h5, "/stage01_fixed/best/equilibrium/psi", np.asarray(eq.psi))
    h5_write_scalar(h5, "/stage01_fixed/best/equilibrium/psi_axis", float(eq.psi_axis))
    h5_write_scalar(h5, "/stage01_fixed/best/equilibrium/psi_lcfs", float(eq.psi_lcfs))
    h5_write_scalar(h5, "/stage01_fixed/best/equilibrium/axis_R", float(eq.axis_R))
    h5_write_scalar(h5, "/stage01_fixed/best/equilibrium/axis_Z", float(eq.axis_Z))

    h5_ensure_group(h5, "/stage01_fixed/best/equilibrium/lcfs")
    h5_write_array(h5, "/stage01_fixed/best/equilibrium/lcfs/R", np.asarray(eq.lcfs_R))
    h5_write_array(h5, "/stage01_fixed/best/equilibrium/lcfs/Z", np.asarray(eq.lcfs_Z))

    h5_write_array(h5, "/stage01_fixed/best/equilibrium/plasma_mask", np.asarray(eq.plasma_mask).astype(np.int8))
    h5_write_array(h5, "/stage01_fixed/best/equilibrium/j_phi", np.asarray(eq.j_phi))

    h5_ensure_group(h5, "/stage01_fixed/best/equilibrium/fields")
    h5_write_array(h5, "/stage01_fixed/best/equilibrium/fields/BR", np.asarray(eq.BR))
    h5_write_array(h5, "/stage01_fixed/best/equilibrium/fields/BZ", np.asarray(eq.BZ))
    h5_write_array(h5, "/stage01_fixed/best/equilibrium/fields/Bphi", np.asarray(eq.Bphi))

    # profiles
    h5_ensure_group(h5, "/stage01_fixed/best/profiles")
    h5_write_array(h5, "/stage01_fixed/best/profiles/psi_bar", np.asarray(eq.psi_bar))
    h5_write_array(h5, "/stage01_fixed/best/profiles/rho", np.asarray(eq.rho))
    h5_write_array(h5, "/stage01_fixed/best/profiles/p", np.asarray(eq.p))
    h5_write_array(h5, "/stage01_fixed/best/profiles/F", np.asarray(eq.F))
    h5_write_array(h5, "/stage01_fixed/best/profiles/Te", np.asarray(eq.Te))
    h5_write_array(h5, "/stage01_fixed/best/profiles/n_e", np.asarray(eq.n_e))
    h5_write_array(h5, "/stage01_fixed/best/profiles/q", np.asarray(eq.q))
    h5_write_array(h5, "/stage01_fixed/best/profiles/s", np.asarray(eq.s))
    h5_write_array(h5, "/stage01_fixed/best/profiles/alpha", np.asarray(eq.alpha))

    # diagnostics
    h5_ensure_group(h5, "/stage01_fixed/best/diagnostics")
    h5_write_scalar(h5, "/stage01_fixed/best/diagnostics/gs_iterations", int(eq.gs_iterations))
    h5_write_scalar(h5, "/stage01_fixed/best/diagnostics/residual_norm", float(eq.residual_norm))