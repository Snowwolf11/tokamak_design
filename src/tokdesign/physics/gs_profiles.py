"""
tokdesign.physics.gs_profiles
=============================

Profile models and normalization helpers used by the fixed-boundary
Grad–Shafranov equilibrium solve.

Stage-01 conventions (from equilibrium_space.yaml)
--------------------------------------------------
Pressure:
    model: "power"
    p(psi_bar) = p0 * (1 - psi_bar)**alpha_p
Toroidal field function:
    model: "linear"
    F(psi_bar) = F0 * (1 - alpha_F * psi_bar)
Enforcement:
    mode: "I_t"
    Scale j_phi so that total toroidal plasma current matches I_t:
        I_t = ∬ j_phi(R,Z) dA

Normalization convention:
    psi_bar = (psi - psi_axis) / (psi_lcfs - psi_axis)
and typically rho := sqrt(psi_bar), but rho is computed elsewhere.

Design notes
------------
This module is deliberately "physics-light but robust":
- It provides stable, monotone profile families that are easy to optimize over.
- It enforces I_t by scaling a chosen current-shape function j_shape(psi_bar).
- It does NOT try to be a self-consistent Grad–Shafranov source model
  using p'(psi) and F F'(psi) directly (that can come later).
  For Stage-01, a controlled shape + correct total current is a good start.

Public API
----------
- normalize_psi(...)
- GSProfiles interface
- build_gs_profiles(cfg_or_controls) -> GSProfiles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import warnings

from tokdesign.constants import E_CHARGE


# =============================================================================
# Public helpers
# =============================================================================

def normalize_psi(
    psi: np.ndarray,
    psi_axis: float,
    psi_lcfs: float,
    *,
    clip: bool = True,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Normalize poloidal flux to psi_bar in [0,1] on the plasma region.

    psi_bar = (psi - psi_axis) / (psi_lcfs - psi_axis)

    Parameters
    ----------
    psi : ndarray
        Poloidal flux on the grid (any shape).
    psi_axis : float
        Flux at magnetic axis (extremum inside LCFS).
    psi_lcfs : float
        Flux on LCFS (Dirichlet boundary).
    clip : bool
        If True, clamp output to [0, 1].
    eps : float
        Small number to avoid divide-by-zero.

    Returns
    -------
    psi_bar : ndarray
        Normalized flux, same shape as psi.
    """
    denom = float(psi_lcfs - psi_axis)
    if abs(denom) < eps:
        # Degenerate normalization; return zeros to avoid NaNs.
        out = np.zeros_like(psi, dtype=float)
        return np.clip(out, 0.0, 1.0) if clip else out

    psi_bar = (np.asarray(psi, dtype=float) - float(psi_axis)) / denom
    if clip:
        psi_bar = np.clip(psi_bar, 0.0, 1.0)
    return psi_bar


# =============================================================================
# GSProfiles interface
# =============================================================================

class GSProfiles(Protocol):
    """
    Minimal interface expected by equilibrium.py.

    NOTE: equilibrium.py currently calls:
        profiles.jphi(RR[mask], ZZ[mask], psi[mask], psi_axis, psi_lcfs)
    i.e. *possibly masked 1D arrays*.
    """

    def jphi(
        self,
        RR: np.ndarray,
        ZZ: np.ndarray,
        psi: np.ndarray,
        psi_axis: float,
        psi_lcfs: float,
    ) -> np.ndarray:
        """Return toroidal current density j_phi (same shape as psi input)."""

    def pressure(self, rho: np.ndarray) -> np.ndarray:
        """Return pressure p(rho) (same shape as rho input)."""

    def toroidal_flux_function(self, rho: np.ndarray) -> np.ndarray:
        """Return F(rho)=R*Bphi (same shape as rho input)."""

    def temperature(self, rho: np.ndarray) -> np.ndarray:
        """Return electron temperature Te(rho) in keV (same shape as rho input)."""

    def density(self, rho: np.ndarray) -> np.ndarray:
        """Return electron density n_e(rho) in m^-3 (same shape as rho input)."""



# =============================================================================
# Profile model implementations
# =============================================================================

@dataclass(frozen=True)
class PressurePowerModel:
    """
    p(psi_bar) = p0 * (1 - psi_bar)**alpha_p
    """
    p0: float
    alpha_p: float

    def __call__(self, psi_bar: np.ndarray) -> np.ndarray:
        psi_bar = np.asarray(psi_bar, dtype=float)
        base = np.clip(1.0 - psi_bar, 0.0, 1.0)
        return float(self.p0) * np.power(base, float(self.alpha_p))


@dataclass(frozen=True)
class TemperaturePowerModel:
    """
    Te(psi_bar) = Te0_keV * (1 - psi_bar)**alpha_Te

    Output is in keV.
    """
    Te0_keV: float
    alpha_Te: float

    def __call__(self, psi_bar: np.ndarray) -> np.ndarray:
        psi_bar = np.asarray(psi_bar, dtype=float)
        base = np.clip(1.0 - psi_bar, 0.0, 1.0)
        Te = float(self.Te0_keV) * np.power(base, float(self.alpha_Te))
        return np.maximum(Te, 0.0)


@dataclass(frozen=True)
class FLinearModel:
    """
    F(psi_bar) = F0 * (1 - alpha_F * psi_bar)
    """
    F0: float
    alpha_F: float

    def __call__(self, psi_bar: np.ndarray) -> np.ndarray:
        psi_bar = np.asarray(psi_bar, dtype=float)
        return float(self.F0) * (1.0 - float(self.alpha_F) * psi_bar)


@dataclass(frozen=True)
class CurrentShapePowerModel:
    """
    A simple non-negative current-shape function in terms of psi_bar:

        j_shape(psi_bar) = (1 - psi_bar)**alpha_j + edge_floor

    This is *not* a "first-principles" GS source term; it is a controlled shape
    that we later scale to enforce total toroidal current I_t.

    Parameters
    ----------
    alpha_j : float
        Peaking exponent (higher -> more peaked on axis).
    edge_floor : float
        Small floor to avoid zero everywhere near LCFS for extreme alpha_j.
    """
    alpha_j: float = 1.5
    edge_floor: float = 0.0

    def __call__(self, psi_bar: np.ndarray) -> np.ndarray:
        psi_bar = np.asarray(psi_bar, dtype=float)
        base = np.clip(1.0 - psi_bar, 0.0, 1.0)
        s = np.power(base, float(self.alpha_j))
        if self.edge_floor != 0.0:
            s = s + float(self.edge_floor)
        # ensure non-negative
        return np.maximum(s, 0.0)


@dataclass
class FixedBoundaryGSProfiles:
    """
    Stage-01 fixed-boundary profile bundle.

    This class:
    - evaluates p and F using the configured models
    - computes a *shape* for j_phi from psi_bar
    - scales j_phi so that ∬ j_phi dA == I_t (if I_t is provided)

    Notes on scaling
    ----------------
    equilibrium.py calls jphi() on masked arrays (only inside LCFS).
    We therefore infer cell area dA from the (RR,ZZ) sample points when possible.
    This works reliably on uniform grids used in Stage-01.
    """
    pressure_model: PressurePowerModel
    F_model: FLinearModel
    j_shape_model: CurrentShapePowerModel

    temperature_model: TemperaturePowerModel
    Ti_over_Te: float = 1.0  # assume Ti = Ti_over_Te * Te

    # enforcement / metadata (may be None if caller didn't pass full controls)
    I_t: Optional[float] = None

    # Optional hints for scaling (if you have them later)
    dR_hint: Optional[float] = None
    dZ_hint: Optional[float] = None

    def pressure(self, rho: np.ndarray) -> np.ndarray:
        # In Stage-01, rho := sqrt(psi_bar). Here we treat rho as input coordinate.
        # Convert back to psi_bar for models defined in psi_bar.
        rho = np.asarray(rho, dtype=float)
        psi_bar = np.clip(rho * rho, 0.0, 1.0)
        return self.pressure_model(psi_bar)

    def toroidal_flux_function(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        psi_bar = np.clip(rho * rho, 0.0, 1.0)
        return self.F_model(psi_bar)

    def jphi(
        self,
        RR: np.ndarray,
        ZZ: np.ndarray,
        psi: np.ndarray,
        psi_axis: float,
        psi_lcfs: float,
    ) -> np.ndarray:
        """
        Compute toroidal current density j_phi.

        If I_t is set:
            scale j_shape so that total current matches I_t.

        If I_t is None:
            return an unscaled "shape current" with an arbitrary magnitude.
            (Not ideal physically, but still allows the PDE to run.)
        """
        RR = np.asarray(RR, dtype=float)
        ZZ = np.asarray(ZZ, dtype=float)
        psi = np.asarray(psi, dtype=float)

        psi_bar = normalize_psi(psi, psi_axis, psi_lcfs, clip=True)
        shape = self.j_shape_model(psi_bar)

        # If no enforcement requested/available, return shape as-is.
        if self.I_t is None or not np.isfinite(float(self.I_t)):
            return shape

        # Infer cell area from RR/ZZ samples.
        dA = _infer_cell_area(RR, ZZ, dR_hint=self.dR_hint, dZ_hint=self.dZ_hint)

        # Total current from shape (discrete approximation): I_shape = sum(shape) * dA
        I_shape = float(np.sum(shape) * dA)
        if not np.isfinite(I_shape) or I_shape <= 0.0:
            # Degenerate: return zeros; solver will likely fail gracefully.
            return np.zeros_like(shape)

        scale = float(self.I_t) / I_shape
        return scale * shape

    def temperature(self, rho: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=float)
        psi_bar = np.clip(rho * rho, 0.0, 1.0)
        return self.temperature_model(psi_bar)


    def density(self, rho: np.ndarray) -> np.ndarray:
        """
        Derive n_e from p and assumed temperatures.

        Assumptions:
        - single ion species with Zi=1 and quasi-neutrality: n_i ~= n_e
        - total pressure p = n_e * (Te + Ti) * e   with Te,Ti in eV
        - Te is modeled; Ti = Ti_over_Te * Te
        """
        p = self.pressure(rho)  # Pa = J/m^3
        Te_keV = self.temperature(rho)

        Ti_over_Te = float(self.Ti_over_Te)
        if not np.isfinite(Ti_over_Te) or Ti_over_Te <= 0.0:
            Ti_over_Te = 1.0

        # Convert keV -> eV -> Joule
        Te_J = Te_keV * 1.0e3 * E_CHARGE
        Ti_J = (Ti_over_Te * Te_keV) * 1.0e3 * E_CHARGE

        denom = Te_J + Ti_J  # J per particle
        denom = np.maximum(denom, 1e-30)

        n_e = np.asarray(p, dtype=float) / denom  # (J/m^3) / (J) = 1/m^3
        return np.maximum(n_e, 0.0)


# =============================================================================
# Factory: build profiles from config / controls
# =============================================================================

def build_gs_profiles(cfg_or_controls: Dict[str, Any]) -> GSProfiles:
    """
    Build Stage-01 profiles.

    Accepts either:
    - the full controls dict (recommended), containing keys:
        controls["profiles"], controls["enforcement"], controls["toroidal_field"], controls["plasma_boundary"]
    - OR only the controls["profiles"] dict (fallback; enforcement defaults).

    Expected Stage-01 variable locations (equilibrium_space.yaml)
    ------------------------------------------------------------
    controls["profiles"]["pressure"]["p0"]
    controls["profiles"]["pressure"]["alpha_p"]
    controls["profiles"]["toroidal_field_function"]["alpha_F"]
    controls["profiles"]["toroidal_field_function"]["F0"]   (often derived)
    controls["toroidal_field"]["B0"]
    controls["toroidal_field"]["R0_B0_ref"]
    controls["enforcement"]["I_t"]
    """
    if not isinstance(cfg_or_controls, dict):
        raise TypeError("build_gs_profiles expects a dict (controls or profiles subtree).")

    # Detect whether this is full controls or just the profiles subtree.
    if "profiles" in cfg_or_controls and isinstance(cfg_or_controls.get("profiles"), dict):
        controls = cfg_or_controls
        profiles_cfg = controls.get("profiles", {}) or {}
    else:
        controls = {}  # not available
        profiles_cfg = cfg_or_controls

    # -------------------------
    # Pressure model
    # -------------------------
    p_cfg = profiles_cfg.get("pressure", {}) if isinstance(profiles_cfg.get("pressure", {}), dict) else {}
    p0 = float(p_cfg.get("p0", 1.0e5))          # default magnitude (Pa-ish)
    alpha_p = float(p_cfg.get("alpha_p", 1.5))  # matches typical Stage-01 init

    pressure_model = PressurePowerModel(p0=p0, alpha_p=alpha_p)

    # -------------------------
    # Temperature model
    # -------------------------
    T_cfg = profiles_cfg.get("temperature", {}) if isinstance(profiles_cfg.get("temperature", {}), dict) else {}
    Te0_keV = float(T_cfg.get("Te0_keV", 10.0))
    alpha_Te = float(T_cfg.get("alpha_Te", 1.0))
    Ti_over_Te = float(T_cfg.get("Ti_over_Te", 1.0))

    temperature_model = TemperaturePowerModel(Te0_keV=Te0_keV, alpha_Te=alpha_Te)

    # -------------------------
    # Toroidal flux function F
    # -------------------------
    F_cfg = (
        profiles_cfg.get("toroidal_field_function", {})
        if isinstance(profiles_cfg.get("toroidal_field_function", {}), dict)
        else {}
    )
    alpha_F = float(F_cfg.get("alpha_F", 0.05))

    # F0: possibly derived from B0 * R0_B0_ref if linkage enabled.
    # (equilibrium_space.yaml has a linkage block for this)
    F0 = None
    if "F0" in F_cfg:
        try:
            F0 = float(F_cfg.get("F0"))
        except Exception:
            F0 = None

    # Try to derive if controls are present and linkages enabled.
    if F0 is None and isinstance(controls, dict) and controls:
        linkages = controls.get("linkages", {}) if isinstance(controls.get("linkages", {}), dict) else {}
        derive = linkages.get("derive_F0_from_B0", {}) if isinstance(linkages.get("derive_F0_from_B0", {}), dict) else {}
        derive_enabled = bool(derive.get("enabled", True))

        if derive_enabled:
            tor_cfg = controls.get("toroidal_field", {}) if isinstance(controls.get("toroidal_field", {}), dict) else {}
            B0 = tor_cfg.get("B0", None)
            Rref = tor_cfg.get("R0_B0_ref", None)
            if B0 is not None and Rref is not None:
                try:
                    F0 = float(B0) * float(Rref)
                except Exception:
                    F0 = None

    # Final fallback if nothing worked:
    if F0 is None:
        # Use a conservative default; this keeps code running but is not "physically tuned".
        F0 = 4.0  # T*m
        warnings.warn(
            "build_gs_profiles: F0 not provided/derivable; using default F0=4.0 (T*m). "
            "Pass full controls with toroidal_field.B0 and toroidal_field.R0_B0_ref to derive F0.",
            RuntimeWarning,
        )

    F_model = FLinearModel(F0=float(F0), alpha_F=alpha_F)

    # -------------------------
    # Current shape + enforcement
    # -------------------------
    # Current model is not explicitly specified in equilibrium_space.yaml v0.1.
    # We choose a sensible default and allow optional overrides:
    #   profiles_cfg["current"] = { "alpha_j": ..., "edge_floor": ... }
    cur_cfg = profiles_cfg.get("current", {}) if isinstance(profiles_cfg.get("current", {}), dict) else {}
    alpha_j = float(cur_cfg.get("alpha_j", max(1.0, alpha_p)))  # tie to pressure peaking by default
    edge_floor = float(cur_cfg.get("edge_floor", 0.0))
    j_shape_model = CurrentShapePowerModel(alpha_j=alpha_j, edge_floor=edge_floor)

    # Enforced total current I_t lives under controls["enforcement"]["I_t"] (Stage-01).
    I_t = None
    if isinstance(controls, dict) and controls:
        enf_cfg = controls.get("enforcement", {}) if isinstance(controls.get("enforcement", {}), dict) else {}
        if "I_t" in enf_cfg:
            try:
                I_t = float(enf_cfg.get("I_t"))
            except Exception:
                I_t = None
    else:
        # If caller passed only profiles subtree, we likely can't enforce.
        # Still build runnable profiles.
        warnings.warn(
            "build_gs_profiles: called without full controls; cannot read enforcement.I_t. "
            "j_phi will be returned unscaled unless you include I_t in the passed dict.",
            RuntimeWarning,
        )

    return FixedBoundaryGSProfiles(
        pressure_model=pressure_model,
        F_model=F_model,
        j_shape_model=j_shape_model,
        temperature_model=temperature_model,
        Ti_over_Te=Ti_over_Te,
        I_t=I_t,
    )


# =============================================================================
# Internals
# =============================================================================

def _infer_cell_area(
    RR: np.ndarray,
    ZZ: np.ndarray,
    *,
    dR_hint: Optional[float] = None,
    dZ_hint: Optional[float] = None,
) -> float:
    """
    Infer cell area dA for uniform rectilinear grids from sample points.

    Works for:
    - full 2D mesh arrays
    - masked 1D arrays (points inside LCFS), as long as there are multiple unique R and Z values

    If inference fails, falls back to 1.0 and warns (so code remains runnable).
    """
    # If hints are supplied, trust them.
    if dR_hint is not None and dZ_hint is not None:
        try:
            dR = float(dR_hint)
            dZ = float(dZ_hint)
            if dR > 0 and dZ > 0:
                return dR * dZ
        except Exception:
            pass

    r = np.asarray(RR, dtype=float).ravel()
    z = np.asarray(ZZ, dtype=float).ravel()

    # Unique coordinates can be expensive; but Stage-01 grid sizes are fine.
    ru = np.unique(r[np.isfinite(r)])
    zu = np.unique(z[np.isfinite(z)])

    dR = None
    dZ = None
    if ru.size >= 2:
        drs = np.diff(np.sort(ru))
        # Use a robust central tendency (median) to avoid edge artifacts.
        dR = float(np.median(drs[drs > 0])) if np.any(drs > 0) else None

    if zu.size >= 2:
        dzs = np.diff(np.sort(zu))
        dZ = float(np.median(dzs[dzs > 0])) if np.any(dzs > 0) else None

    if dR is not None and dZ is not None and dR > 0 and dZ > 0:
        return dR * dZ

    warnings.warn(
        "Could not infer grid cell area dA from RR/ZZ samples; falling back to dA=1.0. "
        "This will break I_t scaling. Provide dR/dZ hints or pass full grid info.",
        RuntimeWarning,
    )
    return 1.0
