#!/usr/bin/env python3
"""
Process_equilibrium.py
======================

Post-processing for a fixed-boundary Grad–Shafranov (GS) equilibrium.

This script is written for the workflow:

  1) Run Grad_Shafranov_solver.py, which saves:
        .../runs/simXXX/gs_solution.h5

  2) Run this script pointing at that file. It will create:
        .../runs/simXXX/postproc/runYYY/

     and store:
       • figures (png + pdf)
       • diagnostics.txt
       • postproc.h5 (derived fields + 1D profiles)

What is computed
----------------
Given (R, Z, ψ) and an inside-mask defining the plasma domain:

A) Derived magnetic field components (axisymmetric):
     BR   = -(1/R) ∂ψ/∂Z
     BZ   =  (1/R) ∂ψ/∂R
     Bphi =  F(ψ)/R         (requires F model consistent with the GS solve)
   and |B|.

B) Magnetic axis location (sub-grid quadratic refinement around ψ minimum).

C) Shafranov shift estimate:
     Δ ≈ R_axis - R0_est
   where R0_est is estimated from the LCFS midplane intercepts.

D) Flux-surface averages vs ψ:
   Flux-surface average is approximated using contour integrals:
     <f>(ψ) ≈ (∮ f dl/|∇ψ|) / (∮ dl/|∇ψ|)

E) q(ψ) proxy using toroidal flux:
     Φ_tor(ψ) = ∬_{region ψ' <= ψ} Bphi dR dZ
     q(ψ) ≈ dΦ_tor/dψ
   Note: The absolute scale depends on your ψ convention and F normalization.
         The *shape* vs ψ is the useful diagnostic at this stage.

Numerical notes
---------------
• Contours are taken from matplotlib's contour algorithm.
• Flux-surface averages are computed on the largest closed contour for each ψ level.
• Toroidal flux is computed by rasterizing the contour polygon onto a subsampled grid.

Dependencies
------------
numpy, matplotlib, scipy, h5py
"""

import os
from pathlib import Path
import h5py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Matplotlib polygon Path for point-in-polygon tests (aliased to avoid clashing with pathlib.Path)
from matplotlib.path import Path as MplPath
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# User-configurable settings
# =============================================================================

# Path to the GS solution you want to analyze (stored by your GS solver)
GS_FILE = "/Users/leon/Desktop/python_skripte/tokamak_design/old/runs/sim001/gs_solution.h5"
MU0 = 4e-7 * np.pi

# These parameters must match the "toy" profile model used in the GS solver,
# at least for F(ψ) because it determines Bphi = F/R.
PROFILE_PARAMS = dict(
    p0=2e4,          # pressure amplitude (toy units)
    alpha=2.0,       # pressure peaking exponent
    F0=1.0,          # F(ψ) normalization (toy units)
    deltaF=0.05,     # fractional reduction of F towards edge
    beta=2.0,        # shaping exponent for F(ψ̄)
)

# Flux surface sampling:
N_PSI_LEVELS = 35
PSI_LEVEL_MARGIN = 0.02  # skip surfaces extremely close to axis and LCFS

# Toroidal flux integration: subsample grid for speed (2 = every 2nd point)
TOR_FLUX_SUBSAMPLE = 2

# Plot styling
FIG_DPI = 120


# =============================================================================
# Helper functions: physics / geometry
# =============================================================================

def find_magnetic_axis(R, Z, psi, inside_mask):
    """
    Locate the magnetic axis and ψ_axis.

    Approach:
      1) Find the minimum ψ within the inside_mask (grid point).
      2) Fit a 2D quadratic surface to a 3×3 stencil around that grid point.
      3) Solve for the stationary point of the quadratic (sub-grid refinement).

    Returns
    -------
    Rax, Zax, psi_axis : floats
        Magnetic axis coordinates and flux at axis.
    """
    psi_inside = np.where(inside_mask, psi, np.inf)
    i0, j0 = np.unravel_index(np.argmin(psi_inside), psi.shape)

    # If the minimum is too close to the grid boundary, we cannot build a 3×3 stencil
    if i0 < 1 or i0 > len(R) - 2 or j0 < 1 or j0 > len(Z) - 2:
        return R[i0], Z[j0], psi[i0, j0]

    # Coordinates of the stencil center
    x0 = R[i0]
    z0 = Z[j0]

    # Fit: f(x,z) = a x^2 + b z^2 + c x z + d x + e z + g
    rows = []
    vals = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = i0 + di
            jj = j0 + dj

            # If stencil leaks outside mask (rare but possible), fall back to grid point
            if not inside_mask[ii, jj]:
                return R[i0], Z[j0], psi[i0, j0]

            x = R[ii] - x0
            z = Z[jj] - z0
            rows.append([x*x, z*z, x*z, x, z, 1.0])
            vals.append(psi[ii, jj])

    A = np.asarray(rows, dtype=float)
    b = np.asarray(vals, dtype=float)
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, bq, c, d, e, g = coeff

    # Stationary point of the quadratic:
    # ∂f/∂x = 2a x + c z + d = 0
    # ∂f/∂z = c x + 2b z + e = 0
    M = np.array([[2*a, c],
                  [c,  2*bq]], dtype=float)
    rhs = -np.array([d, e], dtype=float)

    # If Hessian is ill-conditioned, don't trust the fit
    if np.linalg.cond(M) > 1e10:
        return R[i0], Z[j0], psi[i0, j0]

    x_star, z_star = np.linalg.solve(M, rhs)
    psi_axis = a*x_star*x_star + bq*z_star*z_star + c*x_star*z_star + d*x_star + e*z_star + g
    return x0 + x_star, z0 + z_star, psi_axis


def compute_psibar(psi, psi_axis, psi_b, inside_mask):
    """
    Compute normalized poloidal flux ψ̄ inside LCFS:

      ψ̄ = (ψ - ψ_axis) / (ψ_b - ψ_axis)

    Convention:
      ψ̄ = 0 on axis, ψ̄ = 1 on LCFS.

    Outside the mask we fill ψ̄=1 (not physically meaningful, but convenient).

    Returns
    -------
    psibar : array, same shape as psi
    denom  : float, ψ_b - ψ_axis (used for chain rule scaling)
    """
    denom = (psi_b - psi_axis)
    if abs(denom) < 1e-14:
        denom = np.sign(denom) * 1e-14 if denom != 0 else 1e-14

    psibar = np.full_like(psi, 1.0, dtype=float)
    psibar[inside_mask] = (psi[inside_mask] - psi_axis) / denom
    psibar[inside_mask] = np.clip(psibar[inside_mask], 0.0, 1.0)
    return psibar, denom


def F_of_psibar(psibar, F0, deltaF, beta):
    """
    Toy model for toroidal field function:

      F(ψ̄) = F0 * (1 - deltaF * ψ̄^beta)

    In real equilibria, F(ψ) is a flux function determined by current profile.
    """
    return F0 * (1.0 - deltaF * psibar**beta)


def p_of_psibar(psibar, p0, alpha):
    """
    Toy pressure profile:

      p(ψ̄) = p0 * (1 - ψ̄)^alpha
    """
    return p0 * (1.0 - psibar)**alpha


def polygon_area(Rp, Zp):
    """
    Signed polygon area (shoelace formula).
    Useful to compare contour sizes and pick the largest closed contour.
    """
    return 0.5 * np.sum(Rp[:-1]*Zp[1:] - Rp[1:]*Zp[:-1])


def pick_main_contour(contour_set, level_index):
    """
    For a given contour level, matplotlib may return multiple disjoint closed contours
    (e.g., if there are islands or numerical artifacts).

    We pick the *largest* closed contour by absolute polygon area.

    Returns
    -------
    (Rc, Zc) : closed coordinate arrays (first point repeated at the end)
              or None if no suitable contour exists.
    """
    paths = contour_set.collections[level_index].get_paths()
    if not paths:
        return None

    best = None
    best_area = -np.inf

    for p in paths:
        v = p.vertices
        if v.shape[0] < 20:  # ignore tiny junk contours
            continue

        Rc = v[:, 0]
        Zc = v[:, 1]

        # Close contour explicitly
        Rc2 = np.append(Rc, Rc[0])
        Zc2 = np.append(Zc, Zc[0])

        area = abs(polygon_area(Rc2, Zc2))
        if area > best_area:
            best_area = area
            best = (Rc2, Zc2)

    return best


def estimate_R0_from_lcfs_midplane(R, Z, inside_mask, psi=None, psi_b=None, Rb=None, Zb=None):
    """
    Estimate major radius R0 from LCFS midplane intercepts.

    If the LCFS polygon (Rb,Zb) exists, we use its near-midplane points.
    Otherwise we infer the inboard/outboard intercepts from the inside_mask at Z≈0.

    Returns
    -------
    R0_est : float
        Estimated major radius at midplane: (R_in + R_out)/2
    a_est : float
        Estimated minor radius at midplane: (R_out - R_in)/2
    """
    # Preferred: use boundary polygon if available
    if Rb is not None and Zb is not None:
        idx = np.where(np.abs(Zb) < 1e-3)[0]
        if idx.size >= 2:
            Rin = np.min(Rb[idx])
            Rout = np.max(Rb[idx])
            return 0.5*(Rin + Rout), 0.5*(Rout - Rin)

        # If no exact Z≈0 points exist, use a window around the closest point
        j = np.argmin(np.abs(Zb))
        sl = slice(max(0, j-30), min(len(Zb), j+31))
        Rin = np.min(Rb[sl])
        Rout = np.max(Rb[sl])
        return 0.5*(Rin + Rout), 0.5*(Rout - Rin)

    # Fallback: derive from inside_mask on the Z≈0 row
    j0 = int(np.argmin(np.abs(Z - 0.0)))
    inside_row = inside_mask[:, j0]
    idx = np.where(inside_row)[0]

    if idx.size < 2:
        # Last fallback: use overall R-extent of mask
        Ri = R[np.where(np.any(inside_mask, axis=1))[0]]
        return 0.5*(Ri.min() + Ri.max()), 0.5*(Ri.max() - Ri.min())

    Rin = R[idx.min()]
    Rout = R[idx.max()]
    return 0.5*(Rin + Rout), 0.5*(Rout - Rin)

def q_from_contour(Rc, Zc, interp_F, interp_gradpsi_R, interp_gradpsi_Z):
    """
    Compute q on a given flux surface contour using:
        q = (1/2π) ∮ (F / (R |∇ψ|)) dl

    Rc, Zc are CLOSED arrays (first point repeated at end).
    """
    dRc = np.diff(Rc)
    dZc = np.diff(Zc)
    dl = np.sqrt(dRc**2 + dZc**2)

    Rm = Rc[:-1] + 0.5 * dRc
    Zm = Zc[:-1] + 0.5 * dZc
    pts = np.vstack([Rm, Zm]).T

    Fm = interp_F(pts)
    gR = interp_gradpsi_R(pts)
    gZ = interp_gradpsi_Z(pts)
    gradpsi = np.sqrt(gR**2 + gZ**2)

    integrand = Fm / (Rm * (gradpsi + 1e-30))
    return (1.0 / (2.0 * np.pi)) * np.nansum(integrand * dl)

def trace_fieldline_phi(
    R_start, Z_start,
    interp_BR, interp_BZ, interp_Bphi,
    phi_max=20.0*np.pi,     # total toroidal angle to integrate
    dphi=1e-3,              # step in toroidal angle [rad]
    R_min=None, R_max=None, Z_min=None, Z_max=None
):
    """
    Trace a single magnetic field line using φ as independent variable:
        dR/dφ = R * BR / Bφ
        dZ/dφ = R * BZ / Bφ

    Returns arrays (x,y,z) in Cartesian coordinates.

    Notes:
    - Requires Bφ != 0 (true in tokamaks).
    - Stops early if the integrator leaves bounds or hits NaNs.
    """
    n = int(np.ceil(phi_max / dphi)) + 1
    R = float(R_start)
    Z = float(Z_start)

    phi_arr = np.zeros(n, dtype=float)
    R_arr = np.zeros(n, dtype=float)
    Z_arr = np.zeros(n, dtype=float)

    for k in range(n):
        phi = k * dphi
        phi_arr[k] = phi
        R_arr[k] = R
        Z_arr[k] = Z

        # bounds check (optional)
        if R_min is not None and (R < R_min or R > R_max or Z < Z_min or Z > Z_max):
            phi_arr = phi_arr[:k+1]; R_arr = R_arr[:k+1]; Z_arr = Z_arr[:k+1]
            break

        pt = np.array([[R, Z]], dtype=float)
        BR = float(interp_BR(pt))
        BZ = float(interp_BZ(pt))
        Bphi = float(interp_Bphi(pt))

        if not np.isfinite(BR) or not np.isfinite(BZ) or not np.isfinite(Bphi) or abs(Bphi) < 1e-14:
            phi_arr = phi_arr[:k+1]; R_arr = R_arr[:k+1]; Z_arr = Z_arr[:k+1]
            break

        # Forward Euler step in φ (good enough for visualization)
        dR = R * (BR / Bphi) * dphi
        dZ = R * (BZ / Bphi) * dphi
        R = R + dR
        Z = Z + dZ

    # Convert to Cartesian for 3D plotting
    x = R_arr * np.cos(phi_arr)
    y = R_arr * np.sin(phi_arr)
    z = Z_arr
    return x, y, z


def pick_start_point_on_midplane(R, Z, psi, inside_mask, psi_target):
    """
    Pick a starting point (R_start, Z≈0) on the outboard midplane for a given psi_target.
    Simple method: look at Z=0 row and find R where psi is closest to psi_target.
    """
    j_mid = int(np.argmin(np.abs(Z - 0.0)))
    psi_row = psi[:, j_mid].copy()
    psi_row[~inside_mask[:, j_mid]] = np.nan

    idx = np.nanargmin(np.abs(psi_row - psi_target))
    return float(R[idx]), float(Z[j_mid])


def delta_star_psi(R, Z, psi):
    """
    Compute Δ*ψ = ψ_RR - (1/R) ψ_R + ψ_ZZ on a rectangular (R,Z) grid.
    Assumes R and Z are 1D monotonically increasing arrays.
    """
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    # First derivatives (same conventions as your BR/BZ compute)
    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    # Second derivatives
    d2psi_dR2 = np.gradient(dpsi_dR, dR, axis=0, edge_order=2)
    d2psi_dZ2 = np.gradient(dpsi_dZ, dZ, axis=1, edge_order=2)

    RR = R[:, None]
    return d2psi_dR2 - (1.0 / RR) * dpsi_dR + d2psi_dZ2


def gs_sources_from_psibar(psibar, p0=2e4, F0=1.0, deltaF=0.05):
    """
    Returns p'(ψ) and F F'(ψ) evaluated consistently with your toy model.
    IMPORTANT: This must match the model used in the solver.
    """
    # Clamp to [0,1] like solver does
    x = np.clip(psibar, 0.0, 1.0)

    # Same toy model as described earlier:
    # p(x) = p0*(1-x)^2 -> dp/dx = -2 p0 (1-x)
    dp_dx = -2.0 * p0 * (1.0 - x)

    # F(x) = F0*(1 - deltaF*x) -> dF/dx = -F0*deltaF
    F = F0 * (1.0 - deltaF * x)
    dF_dx = -F0 * deltaF

    # FF' in GS is F dF/dψ. Since we currently have d/dx, we return F dF/dx here.
    # We'll convert to d/dψ below using dx/dψ = 1/(ψ_b - ψ_axis).
    FFprime_x = F * dF_dx

    return dp_dx, FFprime_x


# =============================================================================
# Helper functions: I/O and run-directory management
# =============================================================================

def _next_subdir(parent: Path, prefix: str = "run") -> Path:
    """
    Create a new directory inside `parent` named:

      prefix + XXX   (XXX starts at 001 and increments)

    Example: run001, run002, ...

    Returns
    -------
    d : pathlib.Path
        Newly created directory.
    """
    parent.mkdir(parents=True, exist_ok=True)

    nums = []
    for p in parent.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suf = p.name[len(prefix):]
            if suf.isdigit():
                nums.append(int(suf))

    n = (max(nums) + 1) if nums else 1
    d = parent / f"{prefix}{n:03d}"
    d.mkdir(exist_ok=False)
    return d


def make_postproc_dir_from_solution(gs_file: str, folder_name="postproc", run_prefix="run") -> Path:
    """
    Create a post-processing output directory next to the GS solution file.

    If gs_file is:
      .../runs/sim003/gs_solution.h5

    Create:
      .../runs/sim003/postproc/run001/

    Returns
    -------
    out_dir : pathlib.Path
        Newly created directory for this post-processing run.
    """
    gs_path = Path(gs_file).expanduser().resolve()
    sim_dir = gs_path.parent
    parent = sim_dir / folder_name
    return _next_subdir(parent, prefix=run_prefix)


def load_gs_h5(path: str):
    """
    Load GS solution from an HDF5 file created by Grad_Shafranov_solver.py.

    Returns
    -------
    R : (NR,) array
    Z : (NZ,) array
    psi : (NR,NZ) array
    inside_mask : (NR,NZ) bool array
    Rb, Zb : boundary polygon arrays (or None if not present)
    attrs : dict of file attributes (metadata)
    """
    with h5py.File(path, "r") as f:
        R = f["R"][...]
        Z = f["Z"][...]
        psi = f["psi"][...]
        inside_mask = f["inside_mask"][...].astype(bool)

        Rb = f["Rb"][...] if "Rb" in f else None
        Zb = f["Zb"][...] if "Zb" in f else None

        attrs = dict(f.attrs.items())

    return R, Z, psi, inside_mask, Rb, Zb, attrs


# =============================================================================
# Main routine
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 1) Load GS solution
    # -------------------------------------------------------------------------
    R, Z, psi, inside_mask, Rb, Zb, attrs = load_gs_h5(GS_FILE)

    NR, NZ = psi.shape
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    # ψ boundary value used in your solver (commonly 0.0)
    # If you change this in the solver, update here as well.
    psi_b = 0.0

    # -------------------------------------------------------------------------
    # 2) Axis and normalized flux
    # -------------------------------------------------------------------------
    Rax, Zax, psi_axis = find_magnetic_axis(R, Z, psi, inside_mask)
    psibar, denom = compute_psibar(psi, psi_axis, psi_b, inside_mask)

    # -------------------------------------------------------------------------
    # 3) Profiles (toy model) and magnetic field components
    # -------------------------------------------------------------------------
    # F(ψ̄) -> Bphi = F/R
    F = F_of_psibar(psibar, PROFILE_PARAMS["F0"], PROFILE_PARAMS["deltaF"], PROFILE_PARAMS["beta"])
    # p(ψ̄) optional, mainly for plotting <p>(ψ)
    p = p_of_psibar(psibar, PROFILE_PARAMS["p0"], PROFILE_PARAMS["alpha"])

    # Gradients of ψ on the grid (2nd-order finite differences)
    dpsi_dR = np.gradient(psi, dR, axis=0)
    dpsi_dZ = np.gradient(psi, dZ, axis=1)

    # Axisymmetric field from ψ:
    #   BR = -(1/R) ∂ψ/∂Z
    #   BZ =  (1/R) ∂ψ/∂R
    #   Bphi = F/R
    BR = -(1.0 / RR) * dpsi_dZ
    BZ =  (1.0 / RR) * dpsi_dR
    Bphi = F / RR

    # Magnitude
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2)

    interp_BR = RegularGridInterpolator((R, Z), BR, bounds_error=False, fill_value=np.nan)
    interp_BZ = RegularGridInterpolator((R, Z), BZ, bounds_error=False, fill_value=np.nan)
    interp_Bphi = RegularGridInterpolator((R, Z), Bphi, bounds_error=False, fill_value=np.nan)

    # -------------------------------------------------------------------------
    # 4) Shafranov shift estimate from LCFS midplane geometry
    # -------------------------------------------------------------------------
    R0_est, a_est = estimate_R0_from_lcfs_midplane(R, Z, inside_mask, psi=psi, psi_b=psi_b, Rb=Rb, Zb=Zb)
    shaf_shift = Rax - R0_est

    # -------------------------------------------------------------------------
    # 5) Midplane diagnostics (Z≈0)
    # -------------------------------------------------------------------------
    j_mid = int(np.argmin(np.abs(Z - 0.0)))
    psi_mid = psi[:, j_mid]
    BR_mid = BR[:, j_mid]
    BZ_mid = BZ[:, j_mid]
    Bphi_mid = Bphi[:, j_mid]
    Bmag_mid = Bmag[:, j_mid]

    # -------------------------------------------------------------------------
    # 6) Choose ψ levels for flux surfaces
    # -------------------------------------------------------------------------
    # Typically ψ_axis < ψ_b (axis is minimum). We avoid surfaces too close to axis/edge.
    psi_min = psi_axis + PSI_LEVEL_MARGIN * (psi_b - psi_axis)
    psi_max = psi_b - PSI_LEVEL_MARGIN * (psi_b - psi_axis)
    psi_levels = np.linspace(psi_min, psi_max, N_PSI_LEVELS)

    # Extract contours of ψ at these levels
    # We create the contour set on a temporary figure to avoid displaying it.
    fig_tmp = plt.figure()
    cs = plt.contour(RR, ZZ, psi, levels=psi_levels)
    plt.close(fig_tmp)

     # --- GS residual + jphi diagnostics -----------------------------------------

    # Compute Δ*ψ from the numerically solved ψ
    dstar = delta_star_psi(R, Z, psi)

    # Convert psibar-derivatives to psi-derivatives:
    # psibar = (psi - psi_axis)/(psi_b - psi_axis)  => d/dpsi = (1/(psi_b-psi_axis)) d/dpsibar
    den = (psi_b - psi_axis)
    if abs(den) < 1e-14:
        raise ValueError("psi_b - psi_axis is ~0; cannot form psibar or profile derivatives safely.")

    # Toy model sources in terms of x = psibar
    dp_dx, FFprime_x = gs_sources_from_psibar(
        psibar,
        p0=PROFILE_PARAMS["p0"],
        F0=PROFILE_PARAMS["F0"],
        deltaF=PROFILE_PARAMS["deltaF"],
    )

    # Convert to p'(psi) and (F F')_(psi)
    dp_dpsi = dp_dx / den
    FFprime_psi = FFprime_x / den

    R_col = R[:, None]

    # GS residual: Δ*ψ + μ0 R^2 p'(ψ) + F F'(ψ)
    residual = dstar + MU0 * (R_col**2) * dp_dpsi + FFprime_psi

    # Toroidal current density proxy:
    # From GS: Δ*ψ = - μ0 R j_phi  (sign depends on convention; this matches the usual tokamak convention)
    jphi = -dstar / (MU0 * R_col)

    # Mask outside LCFS for plots/statistics
    dstar_m = np.where(inside_mask, dstar, np.nan)
    resid_m = np.where(inside_mask, residual, np.nan)
    jphi_m  = np.where(inside_mask, jphi, np.nan)

    # Quick scalar diagnostics (helps iteration intuition)
    res_Linf = np.nanmax(np.abs(resid_m))
    res_L2 = np.sqrt(np.nanmean(resid_m**2))

    # Build the source-term magnitude scale for normalization
    term1 = np.abs(dstar)
    term2 = np.abs(MU0 * (R_col**2) * dp_dpsi)   # use your broadcasting column Rcol
    term3 = np.abs(FFprime_psi)
    scale = term1 + term2 + term3 + 1e-30

    Rrel = np.abs(residual) / scale
    Rrel_m = np.where(inside_mask, Rrel, np.nan)

    print("GS relative residual inside LCFS:",
        f"median={np.nanmedian(Rrel_m):.3e},",
        f"p95={np.nanpercentile(Rrel_m,95):.3e},",
        f"max={np.nanmax(Rrel_m):.3e}")

    # Interpolators for evaluating fields along contour points
    # (R, Z) order matches meshgrid indexing="ij".
    interp_Bmag = RegularGridInterpolator((R, Z), Bmag, bounds_error=False, fill_value=np.nan)
    interp_Bphi = RegularGridInterpolator((R, Z), Bphi, bounds_error=False, fill_value=np.nan)
    interp_gradpsi_R = RegularGridInterpolator((R, Z), dpsi_dR, bounds_error=False, fill_value=np.nan)
    interp_gradpsi_Z = RegularGridInterpolator((R, Z), dpsi_dZ, bounds_error=False, fill_value=np.nan)
    interp_F = RegularGridInterpolator((R, Z), F, bounds_error=False, fill_value=np.nan)
    interp_p = RegularGridInterpolator((R, Z), p, bounds_error=False, fill_value=np.nan)
    interp_jphi = RegularGridInterpolator((R, Z), jphi, bounds_error=False, fill_value=np.nan)

    # Arrays to store flux-surface averaged quantities
    fs_psi = []
    fs_Bmag = []
    fs_Bphi = []
    fs_p = []
    fs_q = []
    fs_jphi = []

    # -------------------------------------------------------------------------
    # 7) Toroidal flux Φ_tor(ψ) on a subsampled grid (for q ≈ dΦ/dψ)
    # -------------------------------------------------------------------------
    Rs = R[::TOR_FLUX_SUBSAMPLE]
    Zs = Z[::TOR_FLUX_SUBSAMPLE]
    RRs, ZZs = np.meshgrid(Rs, Zs, indexing="ij")

    # List of points for point-in-polygon queries
    pts_s = np.vstack([RRs.ravel(), ZZs.ravel()]).T

    # Subsampled Bphi for area integration
    Bphi_s = Bphi[::TOR_FLUX_SUBSAMPLE, ::TOR_FLUX_SUBSAMPLE]

    # Cell area on subsampled grid
    cell_area = (Rs[1] - Rs[0]) * (Zs[1] - Zs[0]) if (len(Rs) > 1 and len(Zs) > 1) else dR * dZ

    Phi_tor = []

    # Loop over ψ-levels, build a polygon from the main contour, and compute:
    #  • flux-surface averages
    #  • Φ_tor enclosed by that contour
    for k, lvl in enumerate(psi_levels):
        poly = pick_main_contour(cs, k)
        if poly is None:
            continue

        Rc, Zc = poly
        path = MplPath(np.vstack([Rc, Zc]).T, closed=True)

        # --- Flux-surface averages via weighted contour integral ---
        # Segment midpoints and lengths along the contour
        dRc = np.diff(Rc)
        dZc = np.diff(Zc)
        dl = np.sqrt(dRc**2 + dZc**2)
        Rm = Rc[:-1] + 0.5 * dRc
        Zm = Zc[:-1] + 0.5 * dZc
        seg_pts = np.vstack([Rm, Zm]).T

        # |∇ψ| at segment midpoints
        gR = interp_gradpsi_R(seg_pts)
        gZ = interp_gradpsi_Z(seg_pts)
        gradpsi = np.sqrt(gR**2 + gZ**2)

        # Weight for flux-surface average:
        #   w = dl / |∇ψ|
        w = dl / (gradpsi + 1e-30)

        # Evaluate fields along contour
        Bm = interp_Bmag(seg_pts)
        Bph = interp_Bphi(seg_pts)
        pp = interp_p(seg_pts)

        denom_w = np.nansum(w)
        if denom_w <= 0 or not np.isfinite(denom_w):
            continue
        
        jseg = interp_jphi(seg_pts)

        fs_psi.append(lvl)
        fs_Bmag.append(np.nansum(Bm * w) / denom_w)
        fs_Bphi.append(np.nansum(Bph * w) / denom_w)
        fs_p.append(np.nansum(pp * w) / denom_w)
        q_val = q_from_contour(Rc, Zc, interp_F, interp_gradpsi_R, interp_gradpsi_Z)
        fs_q.append(q_val)
        fs_jphi.append(np.nansum(jseg * w) / denom_w)


        # --- Toroidal flux Φ_tor(ψ): integrate Bphi inside the contour polygon ---
        inside_poly = path.contains_points(pts_s).reshape(RRs.shape)
        Phi = np.sum(Bphi_s[inside_poly]) * cell_area
        Phi_tor.append(Phi)

    # Convert lists to arrays
    fs_psi = np.asarray(fs_psi)
    fs_Bmag = np.asarray(fs_Bmag)
    fs_Bphi = np.asarray(fs_Bphi)
    fs_p = np.asarray(fs_p)
    Phi_tor = np.asarray(Phi_tor)
    fs_q = np.asarray(fs_q)
    fs_jphi = np.asarray(fs_jphi)

    # -------------------------------------------------------------------------
    # 8) q(ψ) estimate: derivative of toroidal flux
    # -------------------------------------------------------------------------
    #if len(fs_psi) >= 3 and len(Phi_tor) == len(fs_psi):
    #    q = np.gradient(Phi_tor, fs_psi)
    #else:
    #    q = np.full_like(fs_psi, np.nan)

    # Convert ψ to normalized ψ̄ for x-axis plots
    psibar_levels = (fs_psi - psi_axis) / (psi_b - psi_axis)
    psibar_levels = np.clip(psibar_levels, 0.0, 1.0)




    # -------------------------------------------------------------------------
    # 9) Create post-processing output directory (next to gs_solution.h5)
    # -------------------------------------------------------------------------
    out_dir = make_postproc_dir_from_solution(GS_FILE, folder_name="postproc", run_prefix="run")
    print(f"Post-processing outputs -> {out_dir}")

    # Convenience: save current figure in both PNG and PDF
    def save_fig(name: str):
        plt.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")

    plt.rcParams["figure.dpi"] = FIG_DPI

    # -------------------------------------------------------------------------
    # 10) Figures
    # -------------------------------------------------------------------------

    # 10.1) ψ contours with LCFS and magnetic axis
    plt.figure(figsize=(7, 6))
    plt.contour(RR, ZZ, psi, levels=40)
    if Rb is not None and Zb is not None:
        plt.plot(Rb, Zb, linewidth=2)
    plt.scatter([Rax], [Zax], s=60, marker="x")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title("Poloidal flux ψ(R,Z) with magnetic axis")
    plt.grid(True, alpha=0.3)
    save_fig("01_psi_contours")

    # 10.2) |B| in poloidal plane
    plt.figure(figsize=(7, 6))
    plt.contourf(RR, ZZ, Bmag, levels=60)
    plt.colorbar(label="|B| [arb]")
    if Rb is not None and Zb is not None:
        plt.plot(Rb, Zb, color="k", linewidth=1.5)
    plt.scatter([Rax], [Zax], s=60, marker="x", color="k")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title("|B|(R,Z)")
    plt.grid(True, alpha=0.2)
    save_fig("02_Bmag_contourf")

    # 10.3) Midplane ψ(R)
    plt.figure(figsize=(8, 5))
    plt.plot(R, psi_mid, label="ψ(R, Z≈0)")
    plt.axvline(Rax, linestyle="--", label="R_axis")
    plt.xlabel("R [m]")
    plt.ylabel("ψ [arb]")
    plt.title("Midplane ψ(R)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig("03_midplane_psi")

    # 10.4) Midplane B components
    plt.figure(figsize=(8, 5))
    plt.plot(R, Bphi_mid, label="Bφ(R, Z≈0)")
    plt.plot(R, BR_mid, label="BR(R, Z≈0)")
    plt.plot(R, BZ_mid, label="BZ(R, Z≈0)")
    plt.plot(R, Bmag_mid, label="|B|(R, Z≈0)")
    plt.axvline(Rax, linestyle="--", label="R_axis")
    plt.xlabel("R [m]")
    plt.ylabel("B [arb]")
    plt.title("Midplane magnetic field components")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig("04_midplane_B_components")

    # 10.5) Flux-surface averages vs ψ̄
    plt.figure(figsize=(8, 5))
    plt.plot(psibar_levels, fs_Bmag, marker="o", label="⟨|B|⟩(ψ)")
    plt.plot(psibar_levels, fs_Bphi, marker="o", label="⟨Bφ⟩(ψ)")
    plt.plot(psibar_levels, fs_p, marker="o", label="⟨p⟩(ψ)")
    plt.xlabel("ψ̄ (normalized poloidal flux)")
    plt.ylabel("Flux-surface average [arb]")
    plt.title("Flux-surface averaged quantities vs ψ̄")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig("05_flux_surface_averages")

    # 10.6) q(ψ̄)
    plt.figure(figsize=(8, 5))
    plt.plot(psibar_levels, fs_q, marker="o", label="q(ψ) from contour integral")
    plt.xlabel("ψ̄ (normalized poloidal flux)")
    plt.ylabel("q [arb units; depends on ψ convention]")
    plt.title("Safety factor proxy from q = dΦ_tor/dψ")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_fig("06_q_profile")

    # Plot maps
    Z2 = Z[None, :]
    R2 = R[:, None]

    fig, ax = plt.subplots()
    im = ax.pcolormesh(RR, ZZ, dstar_m, shading="auto")
    ax.set_title(r"$\Delta^\ast \psi$ (inside LCFS)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_fig("07_GS_operator")

    fig, ax = plt.subplots()
    im = ax.pcolormesh(RR, ZZ, jphi_m, shading="auto")
    ax.set_title(r"$j_\phi \approx -\Delta^\ast\psi/(\mu_0 R)$ (inside LCFS)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_fig("08_j_phi")

    fig, ax = plt.subplots()
    im = ax.pcolormesh(RR, ZZ, resid_m, shading="auto")
    ax.set_title(r"GS residual $\mathcal{R}=\Delta^\ast\psi+\mu_0R^2p' + FF'$ (inside LCFS)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_fig("09_GS_residuals")

    # --- jphi vs R on midplane (Z ~ 0) ------------------------------------------
    jphi_mid = jphi[:, j_mid]  # shape (NR,)

    fig, ax = plt.subplots()
    ax.plot(R, jphi_mid)
    ax.set_title(r"$j_\phi(R)$ on midplane (Z≈0)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel(r"$j_\phi$ [A/m$^2$] (proxy)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("10_j_phi_vs_R")

    fig, ax = plt.subplots()
    ax.plot(psibar_levels, fs_jphi)
    ax.set_title(r"Flux-surface average $\langle j_\phi\rangle(\bar\psi)$")
    ax.set_xlabel(r"Normalized flux $\bar\psi$")
    ax.set_ylabel(r"$\langle j_\phi\rangle$ [A/m$^2$] (proxy)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig("11_j_phi_vs_psi")

    # -------------------------------
    # 3D field line visualization
    # -------------------------------
    # Choose a few starting flux surfaces (normalized), then convert to psi_target
    psi_targets = [
        psi_axis + 0.2*(psi_b - psi_axis),
        psi_axis + 0.5*(psi_b - psi_axis),
        psi_axis + 0.8*(psi_b - psi_axis),
    ]

    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection="3d")

    for i, psi_t in enumerate(psi_targets):
        R0_start, Z0_start = pick_start_point_on_midplane(R, Z, psi, inside_mask, psi_t)
        x, y, z = trace_fieldline_phi(
            R0_start, Z0_start,
            interp_BR, interp_BZ, interp_Bphi,
            phi_max=16*np.pi,    # ~8 toroidal turns
            dphi=2e-3,
            R_min=R.min(), R_max=R.max(), Z_min=Z.min(), Z_max=Z.max()
        )
        ax.plot(x, y, z, linewidth=1.5, label=f"ψ̄≈{(psi_t-psi_axis)/(psi_b-psi_axis):.2f}")

    # --- force equal data scaling in 3D ---
    xs, ys, zs = [], [], []
    for line in ax.lines:
        xd, yd, zd = line.get_data_3d()
        xs.append(xd); ys.append(yd); zs.append(zd)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)

    xmid, ymid, zmid = xs.mean(), ys.mean(), zs.mean()
    max_range = 0.5 * max(xs.ptp(), ys.ptp(), zs.ptp())

    ax.set_title("field_lines")
    ax.set_xlim(xmid - max_range, xmid + max_range)
    ax.set_ylim(ymid - max_range, ymid + max_range)
    ax.set_zlim(zmid - max_range, zmid + max_range)

    # optional: make the axes box cubic too (helps perception)
    ax.set_box_aspect((1, 1, 1))

    # Save (use your existing save_fig pattern, but for 3D we call savefig directly)
    plt.savefig(out_dir / "12_fieldlines_3D.png", dpi=200)
    plt.savefig(out_dir / "12_fieldlines_3D.pdf")


    # -------------------------------------------------------------------------
    # 11) Diagnostics text output
    # -------------------------------------------------------------------------
    diag_text = (
        "\n=== Equilibrium diagnostics (rough) ===\n"
        f"GS file: {GS_FILE}\n"
        f"Magnetic axis: R_axis = {Rax:.6f} m, Z_axis = {Zax:.6f} m\n"
        f"psi_axis = {psi_axis:.6e}, psi_b = {psi_b:.6e}\n"
        f"Estimated midplane R0 = {R0_est:.6f} m, a = {a_est:.6f} m\n"
        f"Shafranov shift estimate Δ = R_axis - R0 ≈ {shaf_shift:.6f} m\n"
        f"Grid: NR={len(R)}, NZ={len(Z)}, dR={dR:.4e} m, dZ={dZ:.4e} m\n"
    )
    print(diag_text)
    with open(out_dir / "diagnostics.txt", "w") as fp:
        fp.write(diag_text)

    # Optional: display figures interactively. Comment out to run headless.
    plt.show()

    # -------------------------------------------------------------------------
    # 12) Save derived data to HDF5
    # -------------------------------------------------------------------------
    out_h5 = out_dir / "postproc.h5"
    with h5py.File(out_h5, "w") as f:
        # Raw fields (copied for convenience so postproc is self-contained)
        f.create_dataset("R", data=R)
        f.create_dataset("Z", data=Z)
        f.create_dataset("psi", data=psi)
        f.create_dataset("inside_mask", data=inside_mask.astype(np.uint8))
        if Rb is not None and Zb is not None:
            f.create_dataset("Rb", data=Rb)
            f.create_dataset("Zb", data=Zb)

        # Derived 2D fields
        f.create_dataset("BR", data=BR)
        f.create_dataset("BZ", data=BZ)
        f.create_dataset("Bphi", data=Bphi)
        f.create_dataset("Bmag", data=Bmag)

        # 1D profiles vs ψ (and ψ̄)
        f.create_dataset("fs_psi", data=fs_psi)
        f.create_dataset("psibar_levels", data=psibar_levels)
        f.create_dataset("q_contour", data=fs_q)
        f.create_dataset("fs_Bmag", data=fs_Bmag)
        f.create_dataset("fs_Bphi", data=fs_Bphi)
        f.create_dataset("fs_p", data=fs_p)
        f.create_dataset("Phi_tor", data=Phi_tor)

        # Scalar diagnostics as attributes
        f.attrs["GS_FILE"] = str(GS_FILE)
        f.attrs["R_axis"] = float(Rax)
        f.attrs["Z_axis"] = float(Zax)
        f.attrs["psi_axis"] = float(psi_axis)
        f.attrs["psi_b"] = float(psi_b)
        f.attrs["R0_est"] = float(R0_est)
        f.attrs["a_est"] = float(a_est)
        f.attrs["shafranov_shift_est"] = float(shaf_shift)
        f.attrs["NR"] = int(len(R))
        f.attrs["NZ"] = int(len(Z))
        f.attrs["dR"] = float(dR)
        f.attrs["dZ"] = float(dZ)

        # --- GS operator / current / residual diagnostics ---------------------------

        grp = f.create_group("gs_diagnostics")

        grp.create_dataset("delta_star_psi", data=dstar)
        grp["delta_star_psi"].attrs["description"] = "Grad-Shafranov operator Δ*ψ"

        grp.create_dataset("jphi", data=jphi)
        grp["jphi"].attrs["description"] = "Toroidal current density proxy j_phi ≈ -Δ*ψ/(μ0 R)"

        grp.create_dataset("gs_residual", data=residual)
        grp["gs_residual"].attrs["description"] = (
            "GS residual R = Δ*ψ + μ0 R^2 p'(ψ) + F(ψ)F'(ψ)"
        )

        # Store some useful scalar norms for quick inspection
        grp.attrs["residual_Linf_inside_LCFS"] = float(res_Linf)
        grp.attrs["residual_L2_inside_LCFS"] = float(res_L2)

        # Store profile parameters used for postproc (helps reproducibility)
        for k, v in PROFILE_PARAMS.items():
            f.attrs[f"profile_{k}"] = float(v)

    print(f"Saved postproc data to: {out_h5}")


if __name__ == "__main__":
    main()
