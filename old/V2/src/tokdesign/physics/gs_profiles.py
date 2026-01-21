"""
gs_profiles.py
==============

Profile models and current-density evaluation for Grad–Shafranov solvers.

This module implements *profile parameterizations* specified by your
target_equilibrium.yaml:

profiles:
  pressure:
    model: "power"
    p0: ...
    alpha_p: ...
  toroidal_field_function:
    model: "linear"
    F0: ...
    alpha_F: ...

Physics
-------
Normalized flux (a.k.a. normalized poloidal flux):
    psin = (psi - psi_axis) / (psi_lcfs - psi_axis)

Analytic v1 profiles:
    p(psin) = p0 * (1 - psin)^alpha_p
    F(psin) = F0 * (1 - alpha_F * psin)

Grad–Shafranov toroidal current density:
    jphi(R,Z) = R * dp/dpsi + (1/(mu0*R)) * F * dF/dpsi

Using chain rule:
    dp/dpsi = (dp/dpsin) * (dpsin/dpsi)
    dF/dpsi = (dF/dpsin) * (dpsin/dpsi)
where:
    dpsin/dpsi = 1 / (psi_lcfs - psi_axis)

Conventions
-----------
- R is strictly > 0 (tokamak convention).
- Arrays:
  * psi: shape (NZ, NR)
  * RR:  shape (NZ, NR) (from np.meshgrid(R, Z, indexing="xy"))

Design notes
------------
- This module is intentionally "dumb": it does not know about LCFS geometry
  beyond the scalar values psi_axis and psi_lcfs used for normalization.
- Enforcement of global constraints (e.g., scaling to hit Ip_target) should
  live in the solver, not here.
- Adding new models later is straightforward: extend the model dispatch in
  PressureProfile / ToroidalFieldFunction.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

MU0 = 4.0e-7 * np.pi  # vacuum permeability [H/m]


# ============================================================
# DATA MODELS (YAML-ALIGNED)
# ============================================================

@dataclass(frozen=True)
class PressureProfile:
    """
    Pressure profile p(psin).

    Supported models (v1)
    ---------------------
    - "power": p(psin) = p0 * (1 - psin)^alpha_p
    """
    model: str
    p0: float
    alpha_p: float

    def p(self, psin: np.ndarray) -> np.ndarray:
        psin = np.asarray(psin, dtype=float)
        if self.model == "power":
            base = np.maximum(0.0, 1.0 - psin)
            return float(self.p0) * base ** float(self.alpha_p)
        raise NotImplementedError(f"pressure.model='{self.model}' not implemented")

    def dp_dpsin(self, psin: np.ndarray) -> np.ndarray:
        """
        dp/dpsin.

        For "power":
            dp/dpsin = -p0 * alpha_p * (1 - psin)^(alpha_p - 1)
        """
        psin = np.asarray(psin, dtype=float)
        if self.model == "power":
            a = float(self.alpha_p)
            if a == 0.0:
                return np.zeros_like(psin)

            base = np.maximum(0.0, 1.0 - psin)

            # Guard base==0 to avoid inf for alpha_p < 1
            out = np.zeros_like(psin)
            m = base > 0.0
            out[m] = -float(self.p0) * a * (base[m] ** (a - 1.0))
            return out

        raise NotImplementedError(f"pressure.model='{self.model}' not implemented")


@dataclass(frozen=True)
class ToroidalFieldFunction:
    """
    Toroidal field function F(psin) = R * Bphi (often called 'F').

    Supported models (v1)
    ---------------------
    - "linear": F(psin) = F0 * (1 - alpha_F * psin)
    """
    model: str
    F0: float
    alpha_F: float

    def F(self, psin: np.ndarray) -> np.ndarray:
        psin = np.asarray(psin, dtype=float)
        if self.model == "linear":
            return float(self.F0) * (1.0 - float(self.alpha_F) * psin)
        raise NotImplementedError(f"toroidal_field_function.model='{self.model}' not implemented")

    def dF_dpsin(self, psin: np.ndarray) -> np.ndarray:
        """
        dF/dpsin.

        For "linear": dF/dpsin = -F0 * alpha_F (constant)
        """
        psin = np.asarray(psin, dtype=float)
        if self.model == "linear":
            return np.full_like(psin, -float(self.F0) * float(self.alpha_F), dtype=float)
        raise NotImplementedError(f"toroidal_field_function.model='{self.model}' not implemented")


@dataclass(frozen=True)
class GSProfiles:
    """
    Bundle of GS profiles: pressure and toroidal field function.
    """
    pressure: PressureProfile
    toroidal_field_function: ToroidalFieldFunction


# ============================================================
# PARSING HELPERS (OPTIONAL BUT HANDY)
# ============================================================

def profiles_from_dict(d: Mapping[str, Any]) -> GSProfiles:
    """
    Construct GSProfiles from a mapping that matches target_equilibrium.yaml.

    Expected keys:
      d["pressure"]["model"], d["pressure"]["p0"], d["pressure"]["alpha_p"]
      d["toroidal_field_function"]["model"], d["toroidal_field_function"]["F0"], d["toroidal_field_function"]["alpha_F"]
    """
    if d is None:
        raise ValueError("profiles mapping is None")

    p = d.get("pressure", {}) or {}
    tf = d.get("toroidal_field_function", {}) or {}

    pressure = PressureProfile(
        model=str(p.get("model", "power")),
        p0=float(p["p0"]),
        alpha_p=float(p["alpha_p"]),
    )

    tor = ToroidalFieldFunction(
        model=str(tf.get("model", "linear")),
        F0=float(tf["F0"]),
        alpha_F=float(tf["alpha_F"]),
    )

    return GSProfiles(pressure=pressure, toroidal_field_function=tor)


# ============================================================
# CORE NUMERICS
# ============================================================

def normalize_psi(
    psi: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalize poloidal flux to psin.

    psin = (psi - psi_axis) / (psi_lcfs - psi_axis)

    Parameters
    ----------
    psi : np.ndarray
        Poloidal flux field, shape (NZ, NR)
    psi_axis : float
        Flux at magnetic axis
    psi_lcfs : float
        Flux at LCFS
    clip : bool
        If True, clip psin to [0,1] to avoid negative bases in power laws.

    Returns
    -------
    psin : np.ndarray
        Normalized flux, same shape as psi
    """
    psi = np.asarray(psi, dtype=float)
    denom = float(psi_lcfs - psi_axis)
    #print(f"psi_axis={psi_axis}, psi_lcfs={psi_lcfs}")
    if abs(denom) < 1e-30:
        raise ValueError("psi_lcfs - psi_axis is too small; cannot normalize psi.")

    psin = (psi - float(psi_axis)) / denom
    if clip:
        psin = np.clip(psin, 0.0, 1.0)
    return psin


def jphi_from_psi(
    psi: np.ndarray,
    RR: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    profiles: GSProfiles,
    *,
    clip_psin: bool = True,
    mu0: float = MU0,
    eps_R: float = 1e-12,
) -> np.ndarray:
    """
    Compute toroidal current density jphi(R,Z) from psi and profile models.

    jphi = R * dp/dpsi + (1/(mu0*R)) * F * dF/dpsi

    Parameters
    ----------
    psi : np.ndarray, shape (NZ, NR)
        Poloidal flux.
    RR : np.ndarray, shape (NZ, NR)
        Radial coordinate mesh [m].
    psi_axis : float
        Flux at magnetic axis.
    psi_lcfs : float
        Flux at LCFS.
    profiles : GSProfiles
        Pressure and toroidal field function models.
    clip_psin : bool
        Clip psin into [0,1] before evaluating profiles (recommended in v1).
    mu0 : float
        Permeability constant.
    eps_R : float
        Small floor for R to avoid division by (near) zero.

    Returns
    -------
    jphi : np.ndarray, shape (NZ, NR)
        Toroidal current density (units depend on conventions for p0 and F0).
    """
    psi = np.asarray(psi, dtype=float)
    RR = np.asarray(RR, dtype=float)
    if psi.shape != RR.shape:
        raise ValueError(f"psi and RR must have the same shape, got {psi.shape} vs {RR.shape}")

    psin = normalize_psi(psi, psi_axis, psi_lcfs, clip=clip_psin)

    denom = float(psi_lcfs - psi_axis)
    inv_denom = 1.0 / denom  # dpsin/dpsi

    dp_dpsin = profiles.pressure.dp_dpsin(psin)
    dp_dpsi = dp_dpsin * inv_denom

    F = profiles.toroidal_field_function.F(psin)
    dF_dpsin = profiles.toroidal_field_function.dF_dpsin(psin)
    dF_dpsi = dF_dpsin * inv_denom

    R_safe = np.maximum(RR, float(eps_R))

    term_p = R_safe * dp_dpsi
    term_F = (F * dF_dpsi) / (float(mu0) * R_safe)

    return term_p + term_F


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing gs_profiles.py")

    # Build a simple grid (conventions match geometry/grids.py)
    R = np.linspace(1.0, 2.0, 12)
    Z = np.linspace(-0.5, 0.5, 11)
    RR, ZZ = np.meshgrid(R, Z, indexing="xy")  # shapes (NZ, NR)

    # Create a psi field such that psi_axis=0, psi_lcfs=1, and psin varies linearly with Z.
    Zmin, Zmax = float(Z[0]), float(Z[-1])
    psin_target = (ZZ - Zmin) / (Zmax - Zmin)  # in [0,1]
    psi = psin_target.copy()

    psi_axis = 0.0
    psi_lcfs = 1.0

    # Profiles matching your YAML structure
    profiles_dict = {
        "pressure": {"model": "power", "p0": 1.2e5, "alpha_p": 1.5},
        "toroidal_field_function": {"model": "linear", "F0": 4.125, "alpha_F": 0.05},
    }
    prof = profiles_from_dict(profiles_dict)

    # 1) Normalization sanity
    psin = normalize_psi(psi, psi_axis, psi_lcfs, clip=True)
    assert np.min(psin) >= -1e-14
    assert np.max(psin) <= 1.0 + 1e-14

    # 2) Model dispatch sanity
    # dp/dpsin analytic check at a few points
    test_vals = np.array([0.0, 0.2, 0.7, 1.0], dtype=float)
    dp_num = prof.pressure.dp_dpsin(test_vals)
    dp_ref = -prof.pressure.p0 * prof.pressure.alpha_p * np.maximum(0.0, 1.0 - test_vals) ** (prof.pressure.alpha_p - 1.0)
    dp_ref[-1] = 0.0  # by our convention at psin=1
    assert np.allclose(dp_num, dp_ref, rtol=1e-12, atol=1e-12)

    dF_num = prof.toroidal_field_function.dF_dpsin(test_vals)
    assert np.allclose(dF_num, -prof.toroidal_field_function.F0 * prof.toroidal_field_function.alpha_F)

    # 3) jphi computation sanity
    jphi = jphi_from_psi(psi, RR, psi_axis, psi_lcfs, prof)
    assert jphi.shape == psi.shape
    assert np.all(np.isfinite(jphi))

    # Should vary with Z (through psin) and with R (through prefactors / 1/R)
    assert float(np.std(jphi)) > 0.0

    # Basic sign sanity: dp/dpsin is negative for alpha_p>0, denom>0 here, so dp/dpsi negative.
    # term_p = R * dp/dpsi should be negative or zero; term_F can be +/- depending on params.
    # Just make sure it's not all zeros.
    assert np.max(np.abs(jphi)) > 0.0

    print("gs_profiles.py self-test passed.")
