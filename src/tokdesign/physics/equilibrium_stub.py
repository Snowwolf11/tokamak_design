"""
tokdesign.physics.equilibrium
=============================

Stub equilibrium generator for Stage 01 wiring.

This is NOT a Grad–Shafranov solver.
It produces a synthetic, analytically defined equilibrium-like dataset on the (R,Z) grid
so that Stage 01 optimization, tracing, HDF5 writing, and plotting can be developed
before the real solver is implemented.

What it does
------------
Given:
  - params: structured dict from controls.x_to_params(x)
  - grid:  RZGrid-like object with R,Z,RR,ZZ,dR,dZ
  - cfg_opt: equilibrium_optimization config (unused except for small defaults)

It constructs:
  - an elliptical LCFS (optionally triangular-ish)
  - a normalized poloidal flux psi such that:
        psi_axis = 0
        psi_lcfs = 1
        psi =  ((R-R0)/a)^2 + (Z/(kappa*a))^2
    so the LCFS is psi=1 and plasma interior is psi<=1.

It then builds:
  - plasma_mask, simple j_phi profile
  - magnetic field components (BR,BZ from psi gradients; Bphi from F/R)
  - 1D profiles p(psi), F(psi), q(rho), shear s, alpha proxy
  - a scalars dict with common metrics (others can be added later)

This gives you a consistent "fake equilibrium" with correct shapes and stable keys.

When replacing with a real GS solver, keep the output contract identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

# Vacuum permeability
MU0 = 4e-7 * np.pi


@dataclass
class EquilibriumResult:
    """
    Output container used by tokdesign.optimization.stage01_fixed.

    stage01_fixed can accept either:
      - dict-like with the same fields
      - or an object with these attributes

    We provide the object form for convenience.
    """
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

    scalars: Dict[str, float]


def solve_fixed_equilibrium(params: Dict[str, Any], grid: Any, cfg_opt: Dict[str, Any]) -> EquilibriumResult:
    """
    Produce a synthetic equilibrium-like solution.

    Parameters
    ----------
    params:
      dict from controls.x_to_params(x), expected keys:
        params["controls"] : nested dict with numeric values
    grid:
      RZGrid-like: has R,Z,RR,ZZ,dR,dZ
    cfg_opt:
      equilibrium_optimization config (unused except for optional profile defaults)

    Returns
    -------
    EquilibriumResult
    """
    controls = params.get("controls", {}) if isinstance(params, dict) else {}
    if not isinstance(controls, dict):
        controls = {}

    # ----------------------------
    # Read a few "controls" (with safe defaults)
    # ----------------------------
    # Major radius and minor radius
    R0 = float(_get(controls, ["plasma_boundary", "R0"], 1.7))
    a = float(_get(controls, ["plasma_boundary", "a"], 0.5))
    Z0 = float(_get(controls, ["plasma_boundary", "Z0"], 0.0))

    # Shaping
    kappa = float(_get(controls, ["plasma_boundary", "kappa"], 1.7))
    delta = float(_get(controls, ["plasma_boundary", "delta"], 0.3))  # triangularity-like

    # Toroidal field reference
    B0 = float(_get(controls, ["toroidal_field", "B0"], 5.0))
    Rref = float(_get(controls, ["toroidal_field", "R0_B0_ref"], R0))
    F0 = float(_get(controls, ["profiles", "toroidal_field_function", "F0"], B0 * Rref))

    # Plasma current (used only for scaling a toy j_phi)
    I_t = float(_get(controls, ["plasma_current", "I_t"], 8e5))

    # Pressure profile controls (toy)
    p0 = float(_get(controls, ["profiles", "pressure", "p0"], 2.0e4))
    p_exp = float(_get(controls, ["profiles", "pressure", "p_exp"], 1.7))

    # q profile endpoints (toy)
    q0_target = float(_get(controls, ["profiles", "safety_factor", "q0"], 1.2))
    q95_target = float(_get(controls, ["profiles", "safety_factor", "q95"], 3.5))

    # ----------------------------
    # Grid arrays
    # ----------------------------
    R = np.asarray(getattr(grid, "R"))
    Z = np.asarray(getattr(grid, "Z"))
    RR = np.asarray(getattr(grid, "RR"))
    ZZ = np.asarray(getattr(grid, "ZZ"))
    dR = float(getattr(grid, "dR"))
    dZ = float(getattr(grid, "dZ"))

    # ----------------------------
    # Define a shaped LCFS curve (ellipse with simple triangularity tweak)
    # ----------------------------
    n_lcfs = 400
    th = np.linspace(0.0, 2.0 * np.pi, n_lcfs, endpoint=False)

    # Triangularity-ish: shift angle in cosine (simple, smooth)
    # R = R0 + a*cos(theta + delta*sin(theta))
    # Z = Z0 + kappa*a*sin(theta)
    lcfs_R = R0 + a * np.cos(th + delta * np.sin(th))
    lcfs_Z = Z0 + (kappa * a) * np.sin(th)

    # ----------------------------
    # Synthetic normalized poloidal flux psi
    # ----------------------------
    # Use an "elliptical radius" rho^2 = ((R-R0)/a)^2 + ((Z-Z0)/(kappa*a))^2
    # Then psi = rho^2 so lcfs is psi=1.
    X = (RR - R0) / max(a, 1e-9)
    Y = (ZZ - Z0) / max(kappa * a, 1e-9)
    psi = X**2 + Y**2

    psi_axis = 0.0
    psi_lcfs = 1.0

    plasma_mask = (psi <= psi_lcfs).astype(np.int8)

    # ----------------------------
    # Toy j_phi (peaked inside plasma)
    # ----------------------------
    # shape it like (1 - psi)^1.5 inside plasma
    # then scale so the total toroidal current is roughly I_t (very rough)
    j_shape = np.zeros_like(psi, dtype=float)
    inside = plasma_mask.astype(bool)
    j_shape[inside] = np.power(np.maximum(1.0 - psi[inside], 0.0), 1.5)

    # Rough scale: choose j0 so that integral(j_phi * dA) ~ I_t/(2*pi*R0)
    # This is not physically correct; it's just stable.
    dA = dR * dZ
    area_int = float(np.sum(j_shape) * dA)
    j0 = 0.0 if area_int <= 0 else (I_t / (2.0 * np.pi * max(R0, 1e-9))) / area_int
    j_phi = j0 * j_shape

    # ----------------------------
    # Magnetic fields
    # ----------------------------
    # For axisymmetry, poloidal field can be derived from psi gradients:
    #   BR ~ -(1/R) dpsi/dZ
    #   BZ ~  (1/R) dpsi/dR
    # Here psi is dimensionless; these are "proxy fields" for wiring.
    dpsi_dZ, dpsi_dR = np.gradient(psi, dZ, dR)  # returns in order (Z, R)
    RR_safe = np.maximum(RR, 1e-6)

    BR = -(1.0 / RR_safe) * dpsi_dZ
    BZ = +(1.0 / RR_safe) * dpsi_dR
    Bphi = F0 / RR_safe

    # ----------------------------
    # 1D profiles on psi_bar grid
    # ----------------------------
    n_psi = int(_get(cfg_opt, ["radial_coordinate", "n_psi"], 201))
    n_psi = max(n_psi, 16)

    psi_bar = np.linspace(0.0, 1.0, n_psi, dtype=float)
    rho = np.sqrt(np.maximum(psi_bar, 0.0))

    # Pressure profile
    p = p0 * np.power(np.maximum(1.0 - psi_bar, 0.0), p_exp)

    # Toroidal field function F(psi) ~ constant in this stub
    F = np.full_like(psi_bar, F0, dtype=float)

    # Safety factor profile: simple monotonic quadratic in rho
    q = q0_target + (q95_target - q0_target) * rho**2

    # Shear s = (rho/q) dq/drho
    dq_drho = np.gradient(q, rho, edge_order=1)
    s = np.zeros_like(q)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (rho / np.maximum(q, 1e-12)) * dq_drho
    s[~np.isfinite(s)] = 0.0

    # Alpha proxy: a very rough ballooning drive proxy from dp/drho and q
    # alpha ~ - (dp/drho) * rho / q^2 (dimensionless-ish)
    dp_drho = np.gradient(p, rho, edge_order=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = -dp_drho * rho / np.maximum(q**2, 1e-12)
    alpha[~np.isfinite(alpha)] = 0.0

    # ----------------------------
    # Diagnostics (toy)
    # ----------------------------
    gs_iterations = 0
    residual_norm = 0.0

    # ----------------------------
    # Scalars for trace metrics
    # ----------------------------
    scalars = _compute_scalars_stub(
        R0=R0, a=a, kappa=kappa, delta=delta, Z0=Z0,
        B0=B0, I_t=I_t, p=p, q=q, rho=rho, alpha=alpha, s=s,
        psi=psi, psi_lcfs=psi_lcfs
    )

    return EquilibriumResult(
        psi=psi.astype(float),
        psi_axis=float(psi_axis),
        psi_lcfs=float(psi_lcfs),
        axis_R=float(R0),
        axis_Z=float(Z0),
        lcfs_R=lcfs_R.astype(float),
        lcfs_Z=lcfs_Z.astype(float),
        plasma_mask=plasma_mask,
        j_phi=j_phi.astype(float),
        BR=BR.astype(float),
        BZ=BZ.astype(float),
        Bphi=Bphi.astype(float),
        psi_bar=psi_bar.astype(float),
        rho=rho.astype(float),
        p=p.astype(float),
        F=F.astype(float),
        q=q.astype(float),
        s=s.astype(float),
        alpha=alpha.astype(float),
        gs_iterations=int(gs_iterations),
        residual_norm=float(residual_norm),
        scalars=scalars,
    )


# =============================================================================
# Helpers
# =============================================================================

def _get(d: Any, path: list, default: Any) -> Any:
    """Safe nested get for dicts."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _compute_scalars_stub(
    *,
    R0: float,
    a: float,
    kappa: float,
    delta: float,
    Z0: float,
    B0: float,
    I_t: float,
    p: np.ndarray,
    q: np.ndarray,
    rho: np.ndarray,
    alpha: np.ndarray,
    s: np.ndarray,
    psi: np.ndarray,
    psi_lcfs: float,
) -> Dict[str, float]:
    """
    Provide a reasonable subset of scalar metrics.
    Many stage01 metrics are advanced and will be computed later; it's fine to omit them.
    Stage01 writer fills missing trace metrics with NaNs.

    Keep the keys aligned with your required schema names where possible.
    """
    # Geometry
    aspect_ratio = R0 / max(a, 1e-9)

    # Rough plasma volume for an axisymmetric elliptical cross-section:
    # Volume ~ 2*pi^2 * R0 * a^2 * kappa
    volume = 2.0 * (np.pi**2) * R0 * (a**2) * kappa

    # Poloidal flux (in our normalized stub it's just psi_lcfs - psi_axis = 1)
    poloidal_flux = float(psi_lcfs)

    # Average pressure (simple average on profile)
    p_avg = float(np.trapz(p, rho) / max((rho[-1] - rho[0]), 1e-12))

    # Stored energy: W ~ (3/2) * <p> * Volume
    stored_energy = 1.5 * p_avg * volume

    # Beta (very rough): beta = 2*mu0*<p> / B0^2
    beta = 2.0 * MU0 * p_avg / max(B0**2, 1e-12)

    # "beta_p" and "beta_N" here are placeholders
    beta_p = beta
    beta_N = beta * 100.0  # totally arbitrary scaling for stub

    # Li placeholder (internal inductance)
    li = 0.8

    # Shape scalars
    kappa_s = float(kappa)
    delta_s = float(delta)

    # Shafranov shift placeholder
    shafranov_shift = 0.0

    # q-related scalars
    q0 = float(q[0])
    q95 = float(q[int(0.95 * (len(q) - 1))])
    q_min = float(np.min(q))
    rho_qmin = float(rho[int(np.argmin(q))])

    # Low-q volume fraction placeholder: fraction of profile points below 1
    low_q_volume_fraction = float(np.mean(q < 1.0))

    # q monotonicity violation: count negative dq/drho segments (normalized)
    dq = np.gradient(q, rho, edge_order=1)
    q_monotonicity_violation = float(np.mean(dq < 0.0))

    # q rational proximity placeholder
    q_rational_proximity = 0.0

    # smoothness placeholders
    q_smoothness = float(np.std(np.gradient(dq, rho, edge_order=1)))

    # shear scalars
    s_edge_mean = float(np.mean(s[int(0.9 * len(s)):]))
    s_edge_min = float(np.min(s[int(0.9 * len(s)):]))
    s_min = float(np.min(s))
    s_max = float(np.max(s))
    negative_shear_extent = float(np.mean(s < 0.0))
    shear_smoothness = float(np.std(np.gradient(s, rho, edge_order=1)))

    # alpha edge
    alpha_edge = alpha[int(0.95 * (len(alpha) - 1)) :]
    alpha_edge_mean = float(np.mean(alpha_edge))
    alpha_edge_p95 = float(np.percentile(alpha_edge, 95))
    alpha_edge_integral = float(np.trapz(np.maximum(alpha_edge, 0.0), rho[int(0.95 * (len(rho) - 1)):]))

    # s-alpha margins (placeholders)
    s_alpha_margin_min = -0.1
    s_alpha_negative_margin_integral = 0.0

    # pressure gradient metrics placeholders
    p_peaking_factor = float(np.max(p) / max(np.mean(p), 1e-12))
    dpdrho = np.gradient(p, rho, edge_order=1)
    dpdrho_max = float(np.max(np.abs(dpdrho)))
    edge_pressure_gradient_integral = float(np.trapz(np.maximum(-dpdrho, 0.0), rho))

    # current metrics placeholders
    j_peaking_factor = 1.5
    current_centroid_shift = 0.0

    return {
        "I_t": float(I_t),
        "B0": float(B0),
        "volume": float(volume),
        "poloidal_flux": float(poloidal_flux),
        "stored_energy": float(stored_energy),
        "aspect_ratio": float(aspect_ratio),
        "beta": float(beta),
        "beta_p": float(beta_p),
        "beta_N": float(beta_N),
        "li": float(li),
        "kappa": float(kappa_s),
        "delta": float(delta_s),
        "shafranov_shift": float(shafranov_shift),

        "q0": float(q0),
        "q95": float(q95),
        "q_min": float(q_min),
        "rho_qmin": float(rho_qmin),
        "low_q_volume_fraction": float(low_q_volume_fraction),
        "q_monotonicity_violation": float(q_monotonicity_violation),
        "q_rational_proximity": float(q_rational_proximity),
        "q_smoothness": float(q_smoothness),

        "s_edge_mean": float(s_edge_mean),
        "s_edge_min": float(s_edge_min),
        "s_min": float(s_min),
        "s_max": float(s_max),
        "negative_shear_extent": float(negative_shear_extent),
        "shear_smoothness": float(shear_smoothness),

        "alpha_edge_mean": float(alpha_edge_mean),
        "alpha_edge_p95": float(alpha_edge_p95),
        "alpha_edge_integral": float(alpha_edge_integral),
        "s_alpha_margin_min": float(s_alpha_margin_min),
        "s_alpha_negative_margin_integral": float(s_alpha_negative_margin_integral),

        "p_peaking_factor": float(p_peaking_factor),
        "dpdrho_max": float(dpdrho_max),
        "edge_pressure_gradient_integral": float(edge_pressure_gradient_integral),
        "j_peaking_factor": float(j_peaking_factor),
        "current_centroid_shift": float(current_centroid_shift),
    }
