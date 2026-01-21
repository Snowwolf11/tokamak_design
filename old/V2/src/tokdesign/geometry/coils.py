"""
coils.py
========

Poloidal Field (PF) coil geometry and basic coil field assembly.

Purpose
-------
This module provides:
• A lightweight PFCoil dataclass (filamentary circular loop approximation)
• Config parsing to create coil objects from baseline_device.yaml
• Utility functions to extract centers/currents and set currents
• Interfaces for coil Green's functions:
    - compute_coil_psi_greens(...) -> G_psi
    - psi_from_coils(...) -> psi_vac

Important separation of concerns
--------------------------------
This module defines coil *geometry* and bookkeeping.

The actual physics for computing psi per ampere from a circular loop is a
separate concern and should live in:
    tokdesign.physics.greens

To keep things modular and testable, compute_coil_psi_greens() calls a function
that you implement in physics/greens.py, e.g.:
    psi_from_filament_loop(RR, ZZ, Rc, Zc, I=1.0)

So:
• geometry/coils.py: coil objects + assembly across coils
• physics/greens.py: single-coil Green's function formulas

Conventions
-----------
• R, Z are in meters
• Currents are in amperes
• Grid arrays RR, ZZ have shape (NZ, NR)
• G_psi has shape (Nc, NZ, NR)

Notes
-----
For early v1, we treat coils as axisymmetric current rings (filaments).
Later extensions can add:
• finite cross-section coils
• multiple filaments per coil
• coil limits / conductor models
• central solenoid multi-turn model
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Any, List, Sequence, Tuple, Optional

import numpy as np


# ============================================================
# DATACLASS
# ============================================================

@dataclass(frozen=True)
class PFCoil:
    """
    Simple PF coil model: filamentary circular loop (axisymmetric ring).

    Attributes
    ----------
    name : str
        Coil name/label (e.g. "PF1U")
    Rc, Zc : float
        Coil center position in R-Z plane [m]
    a : float
        Coil "size proxy" [m]. In v1 this is NOT used in physics (filament model),
        but useful for plotting and clearance constraints.
    I : float
        Coil current [A]
    I_max : float
        Maximum absolute current magnitude [A] (for constraints/optimization)
    """
    name: str
    Rc: float
    Zc: float
    a: float
    I: float
    I_max: float


# ============================================================
# CONFIG PARSING
# ============================================================

def coils_from_config(cfg: Dict[str, Any], *, include_solenoid: bool = True) -> List[PFCoil]:
    """
    Build a list of PFCoil objects from baseline_device.yaml config dict.

    Expected structure:
    -------------------
    pf_coils:
      coils:
        - name: ...
          Rc: ...
          Zc: ...
          a: ...
          I_init: ...
          I_max: ...

    central_solenoid:
      enabled: true/false
      Rc, Zc, a, I_init, I_max, ...

    Parameters
    ----------
    cfg : dict
        Parsed YAML for baseline_device.yaml
    include_solenoid : bool
        If True and central_solenoid.enabled is True, include it as an extra PFCoil.

    Returns
    -------
    coils : list[PFCoil]
    """
    coils: List[PFCoil] = []

    pf = cfg.get("pf_coils", {}) or {}
    pf_list = pf.get("coils", []) or []
    if not isinstance(pf_list, list):
        raise ValueError("pf_coils.coils must be a list of coil definitions.")

    for c in pf_list:
        coils.append(
            PFCoil(
                name=str(c["name"]),
                Rc=float(c["Rc"]),
                Zc=float(c["Zc"]),
                a=float(c.get("a", 0.0)),
                I=float(c.get("I_init", 0.0)),
                I_max=float(c.get("I_max", np.inf)),
            )
        )

    if include_solenoid:
        cs = cfg.get("central_solenoid", {}) or {}
        if bool(cs.get("enabled", False)):
            coils.append(
                PFCoil(
                    name="CS",
                    Rc=float(cs["Rc"]),
                    Zc=float(cs.get("Zc", 0.0)),
                    a=float(cs.get("a", 0.0)),
                    I=float(cs.get("I_init", 0.0)),
                    I_max=float(cs.get("I_max", np.inf)),
                )
            )

    if len(coils) == 0:
        raise ValueError("No coils found in config (pf_coils.coils empty and solenoid disabled).")

    _validate_coils(coils)
    return coils


# ============================================================
# BASIC COIL ARRAY ACCESS
# ============================================================

def coil_centers(coils: Sequence[PFCoil]) -> np.ndarray:
    """
    Return coil centers as array shape (Nc, 2): [[Rc, Zc], ...]
    """
    return np.array([[c.Rc, c.Zc] for c in coils], dtype=float)


def coil_currents(coils: Sequence[PFCoil]) -> np.ndarray:
    """
    Return coil currents as array shape (Nc,).
    """
    return np.array([c.I for c in coils], dtype=float)


def coil_current_limits(coils: Sequence[PFCoil]) -> np.ndarray:
    """
    Return coil current limits as array shape (Nc,) with I_max (absolute).
    """
    return np.array([c.I_max for c in coils], dtype=float)


def set_coil_currents(coils: Sequence[PFCoil], I: np.ndarray) -> List[PFCoil]:
    """
    Return a NEW list of PFCoil objects with updated currents.

    Parameters
    ----------
    coils : sequence[PFCoil]
        Existing coils
    I : np.ndarray, shape (Nc,)
        New currents

    Returns
    -------
    new_coils : list[PFCoil]
        Coils with updated current attribute.
    """
    I = np.asarray(I, dtype=float)
    if I.ndim != 1 or I.shape[0] != len(coils):
        raise ValueError(f"I must have shape ({len(coils)},), got {I.shape}")

    new_coils = [replace(c, I=float(Ii)) for c, Ii in zip(coils, I)]
    return new_coils


# ============================================================
# GREEN'S FUNCTIONS / VACUUM PSI ASSEMBLY
# ============================================================

def compute_coil_psi_greens(
    coils: Sequence[PFCoil],
    RR: np.ndarray,
    ZZ: np.ndarray,
    *,
    method: str = "analytic_elliptic",
) -> np.ndarray:
    """
    Compute psi Green's functions for each coil on the R-Z grid.

    Returns G_psi such that:
        psi_vac(R,Z) = sum_k G_psi[k, :, :] * I_k

    Parameters
    ----------
    coils : sequence[PFCoil]
        Coils (geometry + current values, though only geometry matters here)
    RR, ZZ : np.ndarray, shape (NZ, NR)
        Mesh grids
    method : str
        Which Green's function method to use.
        This is passed to physics.greens implementation.

    Returns
    -------
    G_psi : np.ndarray, shape (Nc, NZ, NR)
        psi per ampere for each coil.
    """
    RR = np.asarray(RR, dtype=float)
    ZZ = np.asarray(ZZ, dtype=float)
    if RR.shape != ZZ.shape or RR.ndim != 2:
        raise ValueError("RR and ZZ must be 2D arrays of identical shape (NZ, NR).")

    # Import here to avoid circular imports and keep geometry module lightweight.
    # You will implement this in tokdesign.physics.greens
    from tokdesign.physics._greens import psi_from_filament_loop

    Nc = len(coils)
    NZ, NR = RR.shape
    G = np.empty((Nc, NZ, NR), dtype=float)

    for k, c in enumerate(coils):
        # psi per 1 ampere
        G[k] = psi_from_filament_loop(RR, ZZ, Rc=c.Rc, Zc=c.Zc, I=1.0, method=method)

    return G


def psi_from_coils(G_psi: np.ndarray, I: np.ndarray) -> np.ndarray:
    """
    Compute vacuum poloidal flux psi from coil Green's functions and currents.

    Parameters
    ----------
    G_psi : np.ndarray, shape (Nc, NZ, NR)
        psi per ampere for each coil
    I : np.ndarray, shape (Nc,)
        coil currents [A]

    Returns
    -------
    psi_vac : np.ndarray, shape (NZ, NR)
        vacuum psi on the grid
    """
    G_psi = np.asarray(G_psi, dtype=float)
    I = np.asarray(I, dtype=float)

    if G_psi.ndim != 3:
        raise ValueError(f"G_psi must have shape (Nc, NZ, NR), got {G_psi.shape}")
    Nc = G_psi.shape[0]
    if I.ndim != 1 or I.shape[0] != Nc:
        raise ValueError(f"I must have shape ({Nc},), got {I.shape}")

    # Efficient contraction: sum_k G[k,:,:] * I[k]
    psi_vac = np.tensordot(I, G_psi, axes=(0, 0))
    # tensordot result shape: (NZ, NR)
    return psi_vac


# ============================================================
# VALIDATION
# ============================================================

def _validate_coils(coils: Sequence[PFCoil]) -> None:
    names = [c.name for c in coils]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate coil names detected: {names}")

    for c in coils:
        if not np.isfinite(c.Rc) or not np.isfinite(c.Zc) or not np.isfinite(c.a) or not np.isfinite(c.I):
            raise ValueError(f"Non-finite coil parameter in {c}")
        if c.Rc <= 0.0:
            raise ValueError(f"Coil Rc must be > 0 (tokamak R-Z). Bad coil: {c}")
        if c.a < 0.0:
            raise ValueError(f"Coil size proxy a must be >= 0. Bad coil: {c}")
        if c.I_max <= 0.0:
            raise ValueError(f"Coil I_max must be > 0. Bad coil: {c}")


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing coils.py")

    # Minimal fake config for PF coils only
    cfg = {
        "pf_coils": {
            "coils": [
                {"name": "PF1U", "Rc": 1.0, "Zc": 1.2, "a": 0.15, "I_init": 0.0, "I_max": 2e6},
                {"name": "PF1L", "Rc": 1.0, "Zc": -1.2, "a": 0.15, "I_init": 0.0, "I_max": 2e6},
            ]
        },
        "central_solenoid": {"enabled": False},
    }

    coils = coils_from_config(cfg)
    print("Coils:", coils)

    centers = coil_centers(coils)
    I = coil_currents(coils)
    Imax = coil_current_limits(coils)
    print("centers:\n", centers)
    print("I:\n", I)
    print("Imax:\n", Imax)

    # Setting currents
    coils2 = set_coil_currents(coils, np.array([1e5, -2e5]))
    print("Updated currents:", coil_currents(coils2))

    # NOTE: compute_coil_psi_greens requires tokdesign.physics.greens.psi_from_filament_loop
    # so we do not call it here to keep the self-test independent.

    print("coils.py self-test passed")
