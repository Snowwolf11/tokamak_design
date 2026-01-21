"""
tokdesign.physics.metrics
========================

Compute *core scalar metrics* for a fixed-boundary Grad–Shafranov equilibrium.

These metrics are intended to populate `EquilibriumResult.scalars` and are
written to HDF5 by `stage01_fixed.py` as required trace metrics.

Design philosophy
-----------------
- This file computes "always useful" and relatively cheap scalars.
- It should not write to disk; Stage 01 orchestrator owns write-out.
- Metrics are computed from arrays already produced by the equilibrium solve:
    ψ, jφ, BR/BZ/Bφ, ψ̄, ρ, p, F, q, s, α
- Accuracy: many quantities here are *engineering approximations* suitable for
  early optimization loops. They are designed to be robust and monotone rather
  than perfect. As the workflow matures, replace approximations with proper
  flux-surface averaging and standard definitions.

Expected inputs (from equilibrium.py caller)
-------------------------------------------
compute_equilibrium_scalars(
    controls=..., grid=..., lcfs_poly=...,
    psi=..., j_phi=..., BR=..., BZ=..., Bphi=...,
    psi_axis=..., psi_lcfs=...,
    psi_bar=..., rho=..., p=..., F=...,
    q=..., s=..., alpha=...,
    gs_iterations=..., residual_norm=...,
)

The function tolerates missing/NaN arrays: it returns NaN for metrics that
cannot be computed safely.

Required scalar names
---------------------
`stage01_fixed.py` expects these in the trace metrics group:
    "I_t","B0","volume","poloidal_flux","stored_energy","aspect_ratio",
    "beta","beta_p","beta_N","li","kappa","delta","shafranov_shift",
    "q0","q95","q_min","rho_qmin","low_q_volume_fraction","q_monotonicity_violation",
    "q_rational_proximity","q_smoothness",
    "s_edge_mean","s_edge_min","s_min","s_max","negative_shear_extent","shear_smoothness",
    "alpha_edge_mean","alpha_edge_p95","alpha_edge_integral","s_alpha_margin_min",
    "s_alpha_negative_margin_integral",
    "p_peaking_factor","dpdrho_max","edge_pressure_gradient_integral",
    "j_peaking_factor","current_centroid_shift",
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from matplotlib.path import Path as MplPath


MU0 = 4e-7 * np.pi


# =============================================================================
# Public API
# =============================================================================

def compute_equilibrium_scalars(**kwargs: Any) -> Dict[str, float]:
    """
    Compute a dictionary of scalar metrics from equilibrium arrays.

    Returns
    -------
    scalars : dict[str, float]
        Keys match what `stage01_fixed.py` expects.
    """
    controls = kwargs.get("controls", {}) or {}
    grid = kwargs.get("grid", None)

    lcfs_poly = np.asarray(kwargs.get("lcfs_poly"), dtype=float)
    psi = np.asarray(kwargs.get("psi"), dtype=float)
    j_phi = np.asarray(kwargs.get("j_phi"), dtype=float)
    BR = np.asarray(kwargs.get("BR"), dtype=float)
    BZ = np.asarray(kwargs.get("BZ"), dtype=float)
    Bphi = np.asarray(kwargs.get("Bphi"), dtype=float)

    psi_axis = float(kwargs.get("psi_axis"))
    psi_lcfs = float(kwargs.get("psi_lcfs"))

    psi_bar = np.asarray(kwargs.get("psi_bar"), dtype=float)
    rho = np.asarray(kwargs.get("rho"), dtype=float)
    p = np.asarray(kwargs.get("p"), dtype=float)

    q = np.asarray(kwargs.get("q"), dtype=float)
    s = np.asarray(kwargs.get("s"), dtype=float)
    alpha = np.asarray(kwargs.get("alpha"), dtype=float)

    gs_iterations = int(kwargs.get("gs_iterations", -1))
    residual_norm = float(kwargs.get("residual_norm", np.nan))

    # --- grid geometry
    RR = np.asarray(getattr(grid, "RR", kwargs.get("RR", None)), dtype=float)
    ZZ = np.asarray(getattr(grid, "ZZ", kwargs.get("ZZ", None)), dtype=float)

    if RR.shape != psi.shape or ZZ.shape != psi.shape:
        # As a fallback, try to reconstruct mesh from 1D arrays if present.
        R1 = getattr(grid, "R", None)
        Z1 = getattr(grid, "Z", None)
        if R1 is None or Z1 is None:
            raise ValueError("compute_equilibrium_scalars: RR/ZZ not provided and grid has no R/Z.")
        R1 = np.asarray(R1, dtype=float)
        Z1 = np.asarray(Z1, dtype=float)
        RR, ZZ = np.meshgrid(R1, Z1)

    dR = float(getattr(grid, "dR", kwargs.get("dR", np.nan)))
    dZ = float(getattr(grid, "dZ", kwargs.get("dZ", np.nan)))
    if not np.isfinite(dR) or not np.isfinite(dZ) or dR <= 0 or dZ <= 0:
        # infer from coordinates
        dR = _infer_spacing(RR[0, :])
        dZ = _infer_spacing(ZZ[:, 0])
    dA = dR * dZ

    # --- plasma mask from LCFS polygon
    lcfs_poly = _ensure_closed_polyline(lcfs_poly)
    plasma_mask = _mask_inside_polyline(RR, ZZ, lcfs_poly)

    # If nothing is inside, return NaNs to avoid crashes downstream.
    if not np.any(plasma_mask):
        return _all_required_nan(gs_iterations=gs_iterations, residual_norm=residual_norm)

    # --- boundary shape metrics from polyline
    geom = _boundary_geometry(lcfs_poly)
    R0_geom = geom["R0"]
    a_geom = geom["a"]
    kappa = geom["kappa"]
    delta = geom["delta"]

    # --- axis location estimate from psi extremum inside plasma
    axis_R, axis_Z = _estimate_axis_RZ(RR, ZZ, psi, plasma_mask)

    # --- volumes / integrals (approximate toroidal volume using 2πR Jacobian)
    R_pl = RR[plasma_mask]
    V = float(np.sum(2.0 * np.pi * R_pl * dA))

    # Area-weighted major radius (volume-weighted with 2πR factor)
    Rbar = float(np.sum(R_pl * (2.0 * np.pi * R_pl * dA)) / (V + 1e-30))

    # Poloidal flux (very simple proxy)
    poloidal_flux = float(abs(psi_lcfs - psi_axis) * 2.0 * np.pi)

    # Pressure + energy
    p_pl = _as_plasma(p, plasma_mask)
    p_vol_avg = float(np.sum(p_pl * (2.0 * np.pi * R_pl * dA)) / (V + 1e-30))
    stored_energy = float(1.5 * np.sum(p_pl * (2.0 * np.pi * R_pl * dA)))  # W = 3/2 ∫ p dV

    # B metrics
    B2 = BR * BR + BZ * BZ + Bphi * Bphi
    Bp2 = BR * BR + BZ * BZ

    B2_pl = _as_plasma(B2, plasma_mask)
    Bp2_pl = _as_plasma(Bp2, plasma_mask)

    B2_vol_avg = float(np.sum(B2_pl * (2.0 * np.pi * R_pl * dA)) / (V + 1e-30))
    Bp2_vol_avg = float(np.sum(Bp2_pl * (2.0 * np.pi * R_pl * dA)) / (V + 1e-30))

    # B0: prefer controls["toroidal_field"]["B0"], else sample Bphi near axis
    B0 = _read_controls_float(controls, ("toroidal_field", "B0"))
    if B0 is None or not np.isfinite(B0):
        B0 = _sample_near_point(Bphi, RR, ZZ, (axis_R, axis_Z), plasma_mask)
    B0 = float(B0) if (B0 is not None and np.isfinite(B0)) else float(np.nan)

    # It: prefer enforcement, else integrate j_phi over cross-section (NOT toroidal volume)
    I_t = _read_controls_float(controls, ("enforcement", "I_t"))
    if I_t is None or not np.isfinite(I_t):
        j_pl = _as_plasma(j_phi, plasma_mask)
        I_t = float(np.sum(j_pl * dA))
    else:
        I_t = float(I_t)

    # Aspect ratio (use boundary-derived a, R0)
    aspect_ratio = float(R0_geom / max(a_geom, 1e-30))

    # Shafranov shift proxy: axis shift relative to geometric center
    shafranov_shift = float(axis_R - R0_geom)

    # Beta definitions (volume averaged)
    beta = float((2.0 * MU0 * p_vol_avg) / (B2_vol_avg + 1e-30))
    beta_p = float((2.0 * MU0 * p_vol_avg) / (Bp2_vol_avg + 1e-30))

    # Normalized beta (common tokamak convention):
    #   beta_N = beta(%) * a(m) * B_T(T) / I_p(MA)
    beta_N = float(100.0 * beta * a_geom * (abs(B0) if np.isfinite(B0) else 0.0) / (abs(I_t) / 1e6 + 1e-30))

    # Internal inductance li (very rough, robust proxy):
    #   li ≈ 2 <Bp^2> / Bp_edge^2
    Bp = np.sqrt(np.maximum(Bp2, 0.0))
    edge_band = plasma_mask & (psi_bar >= 0.9)
    Bp_edge = np.nanpercentile(Bp[edge_band], 90) if np.any(edge_band) else np.nan
    li = float(2.0 * (Bp2_vol_avg) / (Bp_edge * Bp_edge + 1e-30)) if np.isfinite(Bp_edge) else float(np.nan)

    # ---------------- q metrics (computed from q-grid + psi_bar bins)
    q_metrics = _q_metrics(q=q, psi_bar=psi_bar, plasma_mask=plasma_mask, RR=RR, dA=dA)

    # ---------------- shear metrics
    shear_metrics = _band_metrics(
        x=s,
        psi_bar=psi_bar,
        plasma_mask=plasma_mask,
        edge_min=0.8,
        name_prefix="s",
        negative_extent=True,
    )
    shear_smoothness = _profile_smoothness(x=s, psi_bar=psi_bar, plasma_mask=plasma_mask)

    # ---------------- alpha metrics
    alpha_metrics = _alpha_metrics(
    alpha=alpha,
    s=s,
    psi_bar=psi_bar,
    plasma_mask=plasma_mask,
    RR=RR,
    dA=dA,
)

    # ---------------- p metrics
    p_metrics = _peaking_and_gradient_metrics(
        field=p,
        psi_bar=psi_bar,
        plasma_mask=plasma_mask,
        RR=RR,
        dA=dA,
        name_prefix="p",
    )

    # ---------------- j metrics
    j_metrics = _peaking_and_gradient_metrics(
        field=j_phi,
        psi_bar=psi_bar,
        plasma_mask=plasma_mask,
        RR=RR,
        dA=dA,
        name_prefix="j",
        gradient_name=None,  # don't compute dj/drho metrics right now
    )
    current_centroid_shift = _current_centroid_shift(j_phi=j_phi, plasma_mask=plasma_mask, RR=RR, dA=dA, R0_geom=R0_geom)

    # Required dictionary (fill everything explicitly)
    out: Dict[str, float] = {
        "I_t": float(I_t),
        "B0": float(B0),
        "volume": float(V),
        "poloidal_flux": float(poloidal_flux),
        "stored_energy": float(stored_energy),
        "aspect_ratio": float(aspect_ratio),
        "beta": float(beta),
        "beta_p": float(beta_p),
        "beta_N": float(beta_N),
        "li": float(li),
        "kappa": float(kappa),
        "delta": float(delta),
        "shafranov_shift": float(shafranov_shift),

        # q block
        "q0": float(q_metrics["q0"]),
        "q95": float(q_metrics["q95"]),
        "q_min": float(q_metrics["q_min"]),
        "rho_qmin": float(q_metrics["rho_qmin"]),
        "low_q_volume_fraction": float(q_metrics["low_q_volume_fraction"]),
        "q_monotonicity_violation": float(q_metrics["q_monotonicity_violation"]),
        "q_rational_proximity": float(q_metrics["q_rational_proximity"]),
        "q_smoothness": float(q_metrics["q_smoothness"]),

        # s block
        "s_edge_mean": float(shear_metrics["edge_mean"]),
        "s_edge_min": float(shear_metrics["edge_min"]),
        "s_min": float(shear_metrics["min"]),
        "s_max": float(shear_metrics["max"]),
        "negative_shear_extent": float(shear_metrics["negative_extent"]),
        "shear_smoothness": float(shear_smoothness),

        # alpha block
        "alpha_edge_mean": float(alpha_metrics["alpha_edge_mean"]),
        "alpha_edge_p95": float(alpha_metrics["alpha_edge_p95"]),
        "alpha_edge_integral": float(alpha_metrics["alpha_edge_integral"]),
        "s_alpha_margin_min": float(alpha_metrics["s_alpha_margin_min"]),
        "s_alpha_negative_margin_integral": float(alpha_metrics["s_alpha_negative_margin_integral"]),

        # pressure block
        "p_peaking_factor": float(p_metrics["peaking_factor"]),
        "dpdrho_max": float(p_metrics["d_drho_max"]),
        "edge_pressure_gradient_integral": float(p_metrics["edge_grad_integral"]),

        # current block
        "j_peaking_factor": float(j_metrics["peaking_factor"]),
        "current_centroid_shift": float(current_centroid_shift),
    }

    # It’s often useful (debug) to expose solver diagnostics in scalars too.
    # Not required by schema, but harmless and helpful.
    out["gs_iterations"] = float(gs_iterations)
    out["residual_norm"] = float(residual_norm)

    return out


# =============================================================================
# Required-metrics fallback
# =============================================================================

def _all_required_nan(*, gs_iterations: int, residual_norm: float) -> Dict[str, float]:
    required = [
        "I_t","B0","volume","poloidal_flux","stored_energy","aspect_ratio",
        "beta","beta_p","beta_N","li","kappa","delta","shafranov_shift",
        "q0","q95","q_min","rho_qmin","low_q_volume_fraction","q_monotonicity_violation",
        "q_rational_proximity","q_smoothness",
        "s_edge_mean","s_edge_min","s_min","s_max","negative_shear_extent","shear_smoothness",
        "alpha_edge_mean","alpha_edge_p95","alpha_edge_integral","s_alpha_margin_min",
        "s_alpha_negative_margin_integral",
        "p_peaking_factor","dpdrho_max","edge_pressure_gradient_integral",
        "j_peaking_factor","current_centroid_shift",
    ]
    out = {k: float(np.nan) for k in required}
    out["gs_iterations"] = float(gs_iterations)
    out["residual_norm"] = float(residual_norm)
    return out


# =============================================================================
# Geometry / mask helpers
# =============================================================================

def _ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("lcfs_poly must have shape (N,2) with columns [R,Z].")
    if poly.shape[0] < 4:
        raise ValueError("lcfs_poly must have at least 4 points.")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _mask_inside_polyline(RR: np.ndarray, ZZ: np.ndarray, poly: np.ndarray) -> np.ndarray:
    path = MplPath(poly)
    pts = np.column_stack([RR.ravel(), ZZ.ravel()])
    inside = path.contains_points(pts)
    return inside.reshape(RR.shape)


def _infer_spacing(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size < 2:
        return 1.0
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    dx = dx[dx != 0]
    if dx.size == 0:
        return 1.0
    return float(np.median(np.abs(dx)))


def _boundary_geometry(lcfs_poly: np.ndarray) -> Dict[str, float]:
    R = lcfs_poly[:, 0]
    Z = lcfs_poly[:, 1]
    Rmax = float(np.max(R))
    Rmin = float(np.min(R))
    Zmax = float(np.max(Z))
    Zmin = float(np.min(Z))

    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    kappa = (Zmax - Zmin) / (2.0 * max(a, 1e-30))

    # Triangularity: use top and bottom points (closest to Zmax, Zmin)
    i_top = int(np.argmax(Z))
    i_bot = int(np.argmin(Z))
    R_top = float(R[i_top])
    R_bot = float(R[i_bot])

    delta_u = (R0 - R_top) / max(a, 1e-30)
    delta_l = (R0 - R_bot) / max(a, 1e-30)
    delta = 0.5 * (delta_u + delta_l)

    return {"R0": float(R0), "a": float(a), "Z0": float(Z0), "kappa": float(kappa), "delta": float(delta)}


def _estimate_axis_RZ(RR: np.ndarray, ZZ: np.ndarray, psi: np.ndarray, plasma_mask: np.ndarray) -> Tuple[float, float]:
    masked = np.where(plasma_mask, psi, -np.inf)
    flat = int(np.argmax(masked))
    iz, ir = np.unravel_index(flat, psi.shape)
    return float(RR[iz, ir]), float(ZZ[iz, ir])


def _as_plasma(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, float)
    if a.shape != mask.shape:
        # allow scalar
        if a.ndim == 0:
            return np.full(mask.sum(), float(a))
        raise ValueError("Array and mask shape mismatch in _as_plasma.")
    return a[mask]


def _read_controls_float(controls: Dict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
    d: Any = controls
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    try:
        return float(d)
    except Exception:
        return None


def _sample_near_point(
    field: np.ndarray,
    RR: np.ndarray,
    ZZ: np.ndarray,
    pt: Tuple[float, float],
    mask: np.ndarray,
    radius_cells: int = 2,
) -> float:
    """
    Sample median field value in a small neighborhood around (R,Z) point.
    """
    R0, Z0 = pt
    dist2 = (RR - R0) ** 2 + (ZZ - Z0) ** 2
    # choose a threshold based on roughly radius_cells grid cells
    # (we approximate using the median nearest-neighbor distance)
    # If this fails, just take the closest point.
    dR = _infer_spacing(RR[0, :])
    dZ = _infer_spacing(ZZ[:, 0])
    r2 = (radius_cells * dR) ** 2 + (radius_cells * dZ) ** 2
    sel = mask & (dist2 <= r2)
    if np.any(sel):
        return float(np.nanmedian(field[sel]))
    # fallback: closest masked point
    masked = np.where(mask, dist2, np.inf)
    flat = int(np.argmin(masked))
    iz, ir = np.unravel_index(flat, field.shape)
    return float(field[iz, ir])


# =============================================================================
# q / s / alpha metrics
# =============================================================================

def _q_metrics(
    *,
    q: np.ndarray,
    psi_bar: np.ndarray,
    plasma_mask: np.ndarray,
    RR: np.ndarray,
    dA: float,
) -> Dict[str, float]:
    q_pl = q[plasma_mask]
    pb_pl = psi_bar[plasma_mask]
    R_pl = RR[plasma_mask]
    dV_weights = 2.0 * np.pi * R_pl * dA

    if q_pl.size == 0 or not np.any(np.isfinite(q_pl)):
        return dict(
            q0=np.nan, q95=np.nan, q_min=np.nan, rho_qmin=np.nan,
            low_q_volume_fraction=np.nan,
            q_monotonicity_violation=np.nan,
            q_rational_proximity=np.nan,
            q_smoothness=np.nan,
        )

    # q0: median in inner core (psi_bar < 0.02)
    core = pb_pl <= 0.02
    q0 = float(np.nanmedian(q_pl[core])) if np.any(core) else float(np.nanmedian(q_pl[pb_pl <= 0.05]))

    # q95: median around psi_bar ~ 0.95
    edge95 = (pb_pl >= 0.93) & (pb_pl <= 0.97)
    q95 = float(np.nanmedian(q_pl[edge95])) if np.any(edge95) else float(np.nanpercentile(q_pl, 95))

    # q_min, rho_qmin
    q_min = float(np.nanmin(q_pl))
    i_min = int(np.nanargmin(q_pl))
    rho_qmin = float(np.sqrt(np.clip(pb_pl[i_min], 0.0, 1.0)))

    # Low-q volume fraction (q < 1)
    low = q_pl < 1.0
    low_q_vol_frac = float(np.sum(dV_weights[low]) / (np.sum(dV_weights) + 1e-30))

    # 1D binned profile q(pb) for smoothness + monotonicity checks
    pb_c, q_c = _binned_mean(pb_pl, q_pl, nbins=60, pmin=0.0, pmax=1.0)
    dq = np.gradient(q_c, pb_c, edge_order=1)
    # monotonicity violation: fraction of bins with dq < 0 by more than tiny tolerance
    viol = float(np.mean(dq < -1e-4)) if dq.size else float(np.nan)

    # smoothness: RMS of second derivative (dimensionless proxy)
    d2q = np.gradient(dq, pb_c, edge_order=1) if dq.size else np.asarray([])
    smooth = float(np.sqrt(np.nanmean(d2q * d2q))) if d2q.size else float(np.nan)

    # Rational proximity near edge: min distance to common rationals at psi_bar~0.95
    rationals = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    q_edge = float(q95)
    rational_prox = float(np.min(np.abs(q_edge - rationals))) if np.isfinite(q_edge) else float(np.nan)

    return dict(
        q0=q0,
        q95=q95,
        q_min=q_min,
        rho_qmin=rho_qmin,
        low_q_volume_fraction=low_q_vol_frac,
        q_monotonicity_violation=viol,
        q_rational_proximity=rational_prox,
        q_smoothness=smooth,
    )


def _band_metrics(
    *,
    x: np.ndarray,
    psi_bar: np.ndarray,
    plasma_mask: np.ndarray,
    edge_min: float,
    name_prefix: str,
    negative_extent: bool,
) -> Dict[str, float]:
    x_pl = x[plasma_mask]
    pb_pl = psi_bar[plasma_mask]

    out = dict(edge_mean=np.nan, edge_min=np.nan, min=np.nan, max=np.nan, negative_extent=np.nan)
    if x_pl.size == 0 or not np.any(np.isfinite(x_pl)):
        return out

    out["min"] = float(np.nanmin(x_pl))
    out["max"] = float(np.nanmax(x_pl))

    edge = pb_pl >= edge_min
    if np.any(edge):
        out["edge_mean"] = float(np.nanmean(x_pl[edge]))
        out["edge_min"] = float(np.nanmin(x_pl[edge]))

    if negative_extent:
        out["negative_extent"] = float(np.mean(x_pl < 0.0))
    return out


def _profile_smoothness(
    *,
    x: np.ndarray,
    psi_bar: np.ndarray,
    plasma_mask: np.ndarray,
    nbins: int = 60,
) -> float:
    x_pl = x[plasma_mask]
    pb_pl = psi_bar[plasma_mask]
    if x_pl.size == 0 or not np.any(np.isfinite(x_pl)):
        return float(np.nan)

    pb_c, x_c = _binned_mean(pb_pl, x_pl, nbins=nbins, pmin=0.0, pmax=1.0)
    dx = np.gradient(x_c, pb_c, edge_order=1)
    d2x = np.gradient(dx, pb_c, edge_order=1)
    return float(np.sqrt(np.nanmean(d2x * d2x)))


def _alpha_metrics(
    *,
    alpha: np.ndarray,
    s: np.ndarray,
    psi_bar: np.ndarray,
    plasma_mask: np.ndarray,
    RR: np.ndarray,
    dA: float,
    edge_min: float = 0.8,
) -> Dict[str, float]:
    """
    Compute alpha-related metrics, including s–α margin metrics.

    Definitions (Stage-01 proxy)
    ----------------------------
    We use a simple margin proxy:
        margin = s - alpha

    This is not a rigorous ballooning stability boundary, but it is:
    - robust
    - monotone
    - cheap
    - works well as an early constraint signal

    Returned metrics
    ----------------
    alpha_edge_mean:
        Mean(alpha) in edge band (psi_bar >= edge_min).
    alpha_edge_p95:
        95th percentile of alpha in edge band.
    alpha_edge_integral:
        ∫ alpha dV in edge band, with dV = 2π R dA.

    s_alpha_margin_min:
        min(margin) in edge band.

    s_alpha_negative_margin_integral:
        ∫ max(0, -margin) dV in edge band
        (penalizes area where margin is negative, weighted by volume element).
    """
    alpha = np.asarray(alpha, float)
    s = np.asarray(s, float)
    psi_bar = np.asarray(psi_bar, float)
    RR = np.asarray(RR, float)
    plasma_mask = np.asarray(plasma_mask, bool)

    if alpha.shape != psi_bar.shape or s.shape != psi_bar.shape or RR.shape != psi_bar.shape:
        raise ValueError("_alpha_metrics: alpha, s, psi_bar, RR must have the same shape.")

    # Extract plasma values
    a_pl = alpha[plasma_mask]
    s_pl = s[plasma_mask]
    pb_pl = psi_bar[plasma_mask]
    R_pl = RR[plasma_mask]

    # Toroidal volume element weight per cell
    dV = 2.0 * np.pi * R_pl * float(dA)

    out = dict(
        alpha_edge_mean=0.0,
        alpha_edge_p95=0.0,
        alpha_edge_integral=0.0,
        s_alpha_margin_min=0.0,
        s_alpha_negative_margin_integral=0.0,
    )

    # If no plasma points, return safe finite defaults (Stage01 finiteness contract)
    if a_pl.size == 0:
        return out

    # If alpha or s are completely non-finite, return safe zeros
    if not np.any(np.isfinite(a_pl)) or not np.any(np.isfinite(s_pl)) or not np.any(np.isfinite(pb_pl)):
        return out

    # Edge band in plasma
    edge = pb_pl >= float(edge_min)

    # --- alpha edge stats
    if np.any(edge) and np.any(np.isfinite(a_pl[edge])):
        out["alpha_edge_mean"] = float(np.nanmean(a_pl[edge]))
        out["alpha_edge_p95"] = float(np.nanpercentile(a_pl[edge], 95))
        out["alpha_edge_integral"] = float(np.nansum(a_pl[edge] * dV[edge]))

    # --- s-alpha margin stats
    margin_pl = s_pl - a_pl

    if np.any(edge) and np.any(np.isfinite(margin_pl[edge])):
        out["s_alpha_margin_min"] = float(np.nanmin(margin_pl[edge]))

        neg = np.maximum(0.0, -margin_pl[edge])
        out["s_alpha_negative_margin_integral"] = float(np.nansum(neg * dV[edge]))
    elif np.any(np.isfinite(margin_pl)):
        # Fallback: whole plasma if edge band is empty (rare but possible)
        out["s_alpha_margin_min"] = float(np.nanmin(margin_pl))
        neg = np.maximum(0.0, -margin_pl)
        out["s_alpha_negative_margin_integral"] = float(np.nansum(neg * dV))

    # Final safety: never return NaN/inf
    for k, v in list(out.items()):
        if not np.isfinite(v):
            out[k] = 0.0

    return out


# =============================================================================
# Peaking + gradients (p and j)
# =============================================================================

def _peaking_and_gradient_metrics(
    *,
    field: np.ndarray,
    psi_bar: np.ndarray,
    plasma_mask: np.ndarray,
    RR: np.ndarray,
    dA: float,
    name_prefix: str,
    gradient_name: Optional[str] = "d_drho_max",
) -> Dict[str, float]:
    f_pl = field[plasma_mask]
    pb_pl = psi_bar[plasma_mask]
    R_pl = RR[plasma_mask]
    dV = 2.0 * np.pi * R_pl * dA

    out = dict(peaking_factor=np.nan, d_drho_max=np.nan, edge_grad_integral=np.nan)

    if f_pl.size == 0 or not np.any(np.isfinite(f_pl)):
        return out

    # peaking factor: f_axis / <f>_V
    f_vol_avg = float(np.sum(f_pl * dV) / (np.sum(dV) + 1e-30))
    f_axis = float(np.nanpercentile(f_pl[pb_pl <= 0.02], 95)) if np.any(pb_pl <= 0.02) else float(np.nanmax(f_pl))
    out["peaking_factor"] = float(f_axis / (f_vol_avg + 1e-30))

    # Build 1D mean profile f(pb)
    pb_c, f_c = _binned_mean(pb_pl, f_pl, nbins=80, pmin=0.0, pmax=1.0)
    rho_c = np.sqrt(np.clip(pb_c, 0.0, 1.0))

    # d f / d rho
    df_drho = np.gradient(f_c, rho_c + 1e-30, edge_order=1)
    out["d_drho_max"] = float(np.nanmax(np.abs(df_drho)))

    # edge gradient integral: ∫_{rho>0.8} max(0, -df/drho) d rho (proxy)
    edge = rho_c >= 0.8
    if np.any(edge):
        out["edge_grad_integral"] = float(np.trapz(np.maximum(0.0, -df_drho[edge]), rho_c[edge]))

    return out


def _current_centroid_shift(*, j_phi: np.ndarray, plasma_mask: np.ndarray, RR: np.ndarray, dA: float, R0_geom: float) -> float:
    j_pl = j_phi[plasma_mask]
    R_pl = RR[plasma_mask]
    w = np.maximum(j_pl, 0.0) * dA  # cross-section weights for centroid
    if not np.any(w > 0):
        return float(np.nan)
    R_centroid = float(np.sum(R_pl * w) / (np.sum(w) + 1e-30))
    return float(R_centroid - R0_geom)


# =============================================================================
# Binning utilities
# =============================================================================

def _binned_mean(
    p: np.ndarray,
    x: np.ndarray,
    *,
    nbins: int,
    pmin: float,
    pmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean x as a function of p using uniform bins.
    Returns bin centers and mean values, with NaNs for empty bins (filled by interpolation).
    """
    p = np.asarray(p, float)
    x = np.asarray(x, float)

    edges = np.linspace(pmin, pmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full((nbins,), np.nan, dtype=float)

    # digitize into bins
    idx = np.digitize(p, edges) - 1
    ok = (idx >= 0) & (idx < nbins) & np.isfinite(x) & np.isfinite(p)

    for k in range(nbins):
        sel = ok & (idx == k)
        if np.any(sel):
            means[k] = float(np.mean(x[sel]))

    # Fill NaNs by linear interpolation over valid bins
    valid = np.isfinite(means)
    if np.sum(valid) >= 2:
        means = np.interp(centers, centers[valid], means[valid])
    elif np.sum(valid) == 1:
        means[:] = means[valid][0]
    else:
        means[:] = np.nan

    return centers, means
