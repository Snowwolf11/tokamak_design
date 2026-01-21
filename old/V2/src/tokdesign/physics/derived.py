"""
derived.py
==========

Derived quantities from GS equilibrium fields.

This module is used after an equilibrium solve to compute scalar diagnostics:
- Total plasma current Ip
- Plasma cross-sectional area (R–Z plane)
- Internal inductance li (approximate, consistent convention)
- Poloidal beta beta_p (approximate)

Conventions
-----------
- R, Z are 1D arrays (NR,), (NZ,)
- RR is meshgrid, shape (NZ, NR)
- mask is boolean, shape (NZ, NR), True inside plasma region
- jphi is toroidal current density [A/m^2] in the poloidal cross-section

Important notes on conventions
------------------------------
1) Total plasma current:
      Ip = ∬_plasma jphi dR dZ
   (No 2πR factor — jphi is current density through the poloidal cross-section.)

2) Energies / inductance:
   The poloidal magnetic energy in the *full torus* uses dV = 2πR dR dZ.
   We compute:
      Wpol = ∬ (Bp^2/(2μ0)) (2πR) dR dZ
   and use the common dimensionless definition:
      li = 2 Wpol / (μ0 Ip^2)

   This is a useful diagnostic, but still an approximation because:
   - It depends on how you define the plasma region (mask quality).
   - Edge effects / vacuum region are excluded by mask.

3) Poloidal beta (approximate):
   We compute an effective minor radius a_eff from area:
      A = ∬_plasma dR dZ
      a_eff = sqrt(A/π)
   and define a typical edge poloidal field:
      Bp_a = μ0 Ip / (2π a_eff)
   Then:
      beta_p = 2 μ0 <p> / Bp_a^2
   where <p> is the area-average pressure over the plasma region.

   This is a standard engineering approximation suitable for v1 diagnostics.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tokdesign.physics._gs_profiles import normalize_psi

MU0 = 4.0e-7 * np.pi  # [H/m]


def _uniform_spacing(x: np.ndarray, name: str) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 points.")
    if not np.all(np.diff(x) > 0):
        raise ValueError(f"{name} must be strictly increasing.")
    dx_arr = np.diff(x)
    dx = float(np.mean(dx_arr))
    if not np.allclose(dx_arr, dx, rtol=1e-10, atol=0.0):
        raise ValueError(f"{name} grid is not uniform.")
    return dx


def compute_plasma_area(mask: np.ndarray, dR: float, dZ: float) -> float:
    """
    Cross-sectional plasma area in the R–Z plane:
        A = ∬_plasma dR dZ
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")
    return float(np.sum(mask) * float(dR) * float(dZ))


def compute_Ip(
    jphi: np.ndarray,
    dR: float,
    dZ: float,
    mask: np.ndarray,
) -> float:
    """
    Total plasma current:
        Ip = ∬_plasma jphi dR dZ
    """
    jphi = np.asarray(jphi, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if jphi.shape != mask.shape:
        raise ValueError("jphi and mask must have the same shape.")
    return float(np.sum(jphi[mask]) * float(dR) * float(dZ))


def compute_li(
    BR: np.ndarray,
    BZ: np.ndarray,
    RR: np.ndarray,
    mask: np.ndarray,
    dR: float,
    dZ: float,
    Ip: float,
    *,
    mu0: float = MU0,
) -> float:
    """
    Internal inductance li (dimensionless), approximate.

    Steps:
      Bp^2 = BR^2 + BZ^2
      Wpol = ∬ (Bp^2/(2μ0)) (2πR) dR dZ   over plasma
      li   = 2 Wpol / (μ0 Ip^2)

    Returns NaN if Ip is zero/non-finite.
    """
    BR = np.asarray(BR, dtype=float)
    BZ = np.asarray(BZ, dtype=float)
    RR = np.asarray(RR, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if BR.shape != BZ.shape or BR.shape != RR.shape or BR.shape != mask.shape:
        raise ValueError("BR, BZ, RR, mask must all have the same shape.")

    if not np.isfinite(Ip) or abs(float(Ip)) < 1e-30:
        return float("nan")

    Bp2 = BR * BR + BZ * BZ
    integrand = (Bp2 / (2.0 * float(mu0))) * (2.0 * np.pi * RR)

    Wpol = float(np.sum(integrand[mask]) * float(dR) * float(dZ))
    li = (2.0 * Wpol) / (float(mu0) * float(Ip) * float(Ip))
    return float(li)


def compute_beta_p(
    p: np.ndarray,
    mask: np.ndarray,
    dR: float,
    dZ: float,
    Ip: float,
    *,
    mu0: float = MU0,
) -> float:
    """
    Poloidal beta beta_p (engineering approximation):

      A = ∬_plasma dR dZ
      a_eff = sqrt(A/π)
      <p> = (1/A) ∬_plasma p dR dZ
      Bp_a = μ0 Ip / (2π a_eff)
      beta_p = 2 μ0 <p> / Bp_a^2

    Returns NaN if Ip is zero/non-finite or if area is zero.
    """
    p = np.asarray(p, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if p.shape != mask.shape:
        raise ValueError("p and mask must have the same shape.")

    if not np.isfinite(Ip) or abs(float(Ip)) < 1e-30:
        return float("nan")

    A = compute_plasma_area(mask, dR, dZ)
    if A <= 0.0:
        return float("nan")

    p_avg = float(np.sum(p[mask]) * float(dR) * float(dZ) / A)
    a_eff = float(np.sqrt(A / np.pi))

    Bp_a = float(mu0) * float(Ip) / (2.0 * np.pi * a_eff)
    if abs(Bp_a) < 1e-30:
        return float("nan")

    beta_p = (2.0 * float(mu0) * p_avg) / (Bp_a * Bp_a)
    return float(beta_p)


def approx_q_profile(
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    BR: np.ndarray,
    BZ: np.ndarray,
    profiles,  # GSProfiles from gs_profiles.py
    psi_axis: float,
    psi_lcfs: float,
    axis_RZ: Tuple[float, float],
    *,
    psin_levels: Optional[np.ndarray] = None,
    n_levels: int = 25,
    psin_min: float = 0.05,
    psin_max: float = 0.95,
    min_points: int = 40,
    eps_Bp: float = 1e-12,
) -> Dict[str, np.ndarray]:
    """
    Compute an approximate q-profile q(psin) by integrating along flux-surface contours.

    Uses:
        q = (1/(2π)) ∮ (Bphi / (R*Bp)) dl_p
    with:
        Bp = sqrt(BR^2 + BZ^2)
        Bphi = F(psin)/R

    Parameters
    ----------
    R, Z : 1D arrays
        Grid coordinates.
    psi : 2D array (NZ, NR)
        Poloidal flux.
    BR, BZ : 2D arrays (NZ, NR)
        Poloidal magnetic field components from fields.py.
    profiles : GSProfiles
        Provides F(psin) via profiles.toroidal_field_function.F(psin).
    psi_axis, psi_lcfs : float
        Flux values at axis and LCFS for normalization.
    axis_RZ : (Raxis, Zaxis)
        Magnetic axis location (used to pick correct closed contour).
    psin_levels : optional array
        Explicit normalized-flux levels in (0,1).
    n_levels, psin_min, psin_max : controls if psin_levels not provided
    min_points : reject tiny/degenerate contours
    eps_Bp : floor to avoid division by ~0

    Returns
    -------
    dict with:
      - "psin": array of levels successfully computed
      - "q":    array of q values (same length)
    """
    import matplotlib
    matplotlib.use("Agg")  # safe for scripts
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from scipy.interpolate import RegularGridInterpolator

    R = np.asarray(R, float)
    Z = np.asarray(Z, float)
    psi = np.asarray(psi, float)
    BR = np.asarray(BR, float)
    BZ = np.asarray(BZ, float)

    NZ, NR = psi.shape
    if (NZ, NR) != (Z.size, R.size):
        raise ValueError("psi shape must be (NZ,NR) matching Z and R.")
    if BR.shape != psi.shape or BZ.shape != psi.shape:
        raise ValueError("BR/BZ must have same shape as psi.")

    # Normalized flux field
    psin_field = normalize_psi(psi, psi_axis, psi_lcfs, clip=False)

    # Decide levels
    if psin_levels is None:
        psin_levels = np.linspace(psin_min, psin_max, n_levels, dtype=float)
    else:
        psin_levels = np.asarray(psin_levels, float)

    # Interpolators for Bp and for R (R is 1D but we sample at points)
    Bp = np.sqrt(BR * BR + BZ * BZ)
    Bp_itp = RegularGridInterpolator((Z, R), Bp, bounds_error=False, fill_value=np.nan)

    # For convenience we just use the x-coordinate directly as R along contour points.
    # (Contour points are already in physical (R,Z).)

    # Determine axis-inclusion test
    Rax, Zax = float(axis_RZ[0]), float(axis_RZ[1])
    axis_point = np.array([Rax, Zax], dtype=float)

    psin_out = []
    q_out = []

    # Create contours in one go using a throwaway figure (Agg backend)
    fig = plt.figure()
    try:
        cs = plt.contour(R, Z, psin_field, levels=psin_levels)
        for lev, segs in zip(cs.levels, cs.allsegs):
            if segs is None or len(segs) == 0:
                continue

            # Pick the segment that:
            #  (a) has enough points
            #  (b) is closed (start≈end) or can be treated as closed
            #  (c) encloses the axis point
            chosen = None
            for seg in segs:
                if seg.shape[0] < min_points:
                    continue

                # Ensure closure for integration
                if not np.allclose(seg[0], seg[-1]):
                    seg = np.vstack([seg, seg[0]])

                # Check if it encloses axis
                if Path(seg).contains_point(axis_point):
                    chosen = seg
                    break

            if chosen is None:
                continue

            # Arc-length along contour
            dR = np.diff(chosen[:, 0])
            dZ = np.diff(chosen[:, 1])
            dl = np.sqrt(dR * dR + dZ * dZ)  # length of each segment
            # Midpoints for sampling fields
            Rm = 0.5 * (chosen[:-1, 0] + chosen[1:, 0])
            Zm = 0.5 * (chosen[:-1, 1] + chosen[1:, 1])

            # Sample Bp at midpoints
            pts = np.column_stack([Zm, Rm])  # interpolator expects (Z,R)
            Bp_m = Bp_itp(pts)

            # Reject if interpolation failed
            if not np.all(np.isfinite(Bp_m)):
                continue

            # Toroidal field function at this surface (depends only on psin level)
            F_lev = float(profiles.toroidal_field_function.F(np.array([lev], float))[0])

            # Bphi = F/R
            Rm_safe = np.maximum(Rm, 1e-12)
            Bphi_m = F_lev / Rm_safe

            # Integrand: (Bphi / (R*Bp)) dl
            Bp_safe = np.maximum(Bp_m, eps_Bp)
            integrand = (Bphi_m / (Rm_safe * Bp_safe)) * dl
            dphi = float(np.sum(integrand))
            q = dphi / (2.0 * np.pi)

            psin_out.append(float(lev))
            q_out.append(float(q))
    finally:
        plt.close(fig)

    return {
        "psin": np.asarray(psin_out, dtype=float),
        "q": np.asarray(q_out, dtype=float),
    }



# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing derived.py (including q-profile)")

    import numpy as np

    from tokdesign.geometry.grid import make_rz_grid
    from tokdesign.physics._fields import compute_BR_BZ_from_psi
    from tokdesign.physics._gs_profiles import GSProfiles, PressureProfile, ToroidalFieldFunction, normalize_psi
    # IMPORTANT: approx_q_profile must be implemented in this derived.py (as provided earlier)

    # ------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------
    R0 = 2.0
    a_lcfs = 0.6

    R, Z, RR, ZZ = make_rz_grid(
        R_min=R0 - 0.9,
        R_max=R0 + 0.9,
        Z_min=-0.9,
        Z_max=0.9,
        NR=240,
        NZ=240,
    )

    # Uniform spacings (reuse internal helper if present)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    # ------------------------------------------------------------
    # Manufactured equilibrium:
    # psi = (R-R0)^2 + Z^2  -> circular flux surfaces
    # ------------------------------------------------------------
    psi = (RR - R0) ** 2 + ZZ ** 2
    psi_axis = 0.0
    psi_lcfs = a_lcfs ** 2

    # Poloidal field from psi
    BR, BZ = compute_BR_BZ_from_psi(R, Z, psi)

    # ------------------------------------------------------------
    # Profiles: constant F = F0 (alpha_F=0)
    # Pressure is irrelevant for q, but GSProfiles requires it.
    # ------------------------------------------------------------
    F0 = 4.0  # [T*m], arbitrary but fixed
    profiles = GSProfiles(
        pressure=PressureProfile(model="power", p0=0.0, alpha_p=1.0),
        toroidal_field_function=ToroidalFieldFunction(model="linear", F0=F0, alpha_F=0.0),
    )

    axis_RZ = (R0, 0.0)

    # ------------------------------------------------------------
    # Validate q(psin) vs analytic q(r) for several psin levels
    # psin = psi/psi_lcfs = r^2/a_lcfs^2 -> r = a_lcfs*sqrt(psin)
    # analytic: q(r) = F0 / (2*sqrt(R0^2 - r^2))
    # ------------------------------------------------------------
    psin_levels = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=float)

    out = approx_q_profile(
        R=R,
        Z=Z,
        psi=psi,
        BR=BR,
        BZ=BZ,
        profiles=profiles,
        psi_axis=psi_axis,
        psi_lcfs=psi_lcfs,
        axis_RZ=axis_RZ,
        psin_levels=psin_levels,
        min_points=120,     # more points -> more accurate integral
        eps_Bp=1e-12,
    )

    psin_got = out["psin"]
    q_got = out["q"]

    # Ensure we computed all requested levels (contours should exist for these)
    assert psin_got.size == psin_levels.size, (
        f"Expected {psin_levels.size} q points, got {psin_got.size}. "
        "Contour extraction failed for some levels."
    )

    # Analytic reference
    r = a_lcfs * np.sqrt(psin_levels)
    q_ref = F0 / (2.0 * np.sqrt(R0 * R0 - r * r))

    # Compare
    rel_err = np.abs(q_got - q_ref) / np.maximum(1e-14, np.abs(q_ref))

    print("  psin:", psin_levels)
    print("  q_ref:", q_ref)
    print("  q_got:", q_got)
    print("  rel_err:", rel_err)

    # Tolerance: contour extraction + interpolation introduces small errors.
    # With a dense grid and enough contour points, this should be quite tight.
    assert np.max(rel_err) < 5e-3, "q-profile test failed: relative error too large."

    # ------------------------------------------------------------
    # Keep existing basic tests for Ip/beta_p/li (sanity)
    # ------------------------------------------------------------
    # Simple rectangular mask, constant jphi -> Ip = j0 * area
    mask = (RR > (R0 - 0.2)) & (RR < (R0 + 0.2)) & (np.abs(ZZ) < 0.2)
    assert np.any(mask)

    j0 = 3.0e6
    jphi = np.zeros_like(RR)
    jphi[mask] = j0

    A = compute_plasma_area(mask, dR, dZ)
    Ip = compute_Ip(jphi, dR, dZ, mask)
    Ip_ref = j0 * A
    assert abs(Ip - Ip_ref) / max(1.0, abs(Ip_ref)) < 1e-12

    p0 = 2.0e5
    p = np.zeros_like(RR)
    p[mask] = p0
    beta_p = compute_beta_p(p, mask, dR, dZ, Ip)
    assert np.isfinite(beta_p) and beta_p > 0.0

    # Internal inductance sanity (not physically consistent field, but should be finite/positive)
    BRt = np.zeros_like(RR)
    BZt = np.zeros_like(RR)
    BZt[mask] = 0.5
    li = compute_li(BRt, BZt, RR, mask, dR, dZ, Ip)
    assert np.isfinite(li) and li > 0.0

    print("derived.py self-test passed (including q-profile).")

