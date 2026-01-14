#!/usr/bin/env python3
"""
Free-boundary GS with force balance (step 3; no X-point)
========================================================

Solve:
    Δ*psi = - μ0 R^2 p'(psi) - F(psi) F'(psi)

Boundary condition:
    psi = psi_ext from PF coils on the outer box boundary.

Plasma region:
    defined by psibar <= 1, where psibar = (psi-psi_axis)/(psi_sep-psi_axis)
    and psi_sep is taken from a reference point (R_ref, Z_ref).  (No X-point.)

Profiles (defaults, overridable from CSV):
    p(psibar) = p0 * (1 - psibar)^alpha_p
    F(psibar) = F0 * (1 - deltaF * psibar^betaF)

Inputs:
  - PARAMS_CSV (key,value)
  - COILS_CSV (name,Rc,Zc,I)

Outputs:
  HARD_CODED_BASE/gs_force_balanceXXX/
      equilibrium.h5
      plots/*.png
"""

import os
import re
import glob
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.special import ellipk, ellipe
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7 * np.pi

# -------------------------- hardcoded paths ---------------------------
HARD_CODED_BASE = "/Users/leon/Desktop/python_skripte/tokamak_design/runs"

PARAMS_CSV = "/Users/leon/Desktop/python_skripte/tokamak_design/inputs/gs_force_balance_params.csv"
COILS_CSV  = "/Users/leon/Desktop/python_skripte/tokamak_design/inputs/pf_coils.csv"

OUT_PREFIX = "gs_force_balance"
H5_NAME = "equilibrium.h5"

# -------------------------- defaults ---------------------------

DEFAULTS = dict(
    # domain grid
    Rmin=0.7, Rmax=3.4, Zmin=-2.6, Zmax=2.6,
    NR=257, NZ=257,

    # TF model for plotting (does not enter GS)
    TF_R0=1.65,
    TF_B0=2.5,

    # step-3 edge definition (no X-point): psi_sep = psi(R_ref, Z_ref)
    R_ref=2.2,
    Z_ref=0.0,

    # profile parameters
    p0=2.0e4,        # [Pa] (toy but physical units)
    alpha_p=1.5,     # peaking exponent
    F0=6.0,          # [T*m] (since Bphi=F/R)
    deltaF=0.12,     # fractional drop from axis to edge
    betaF=1.0,       # shaping exponent

    # iteration controls
    max_iter=60,
    tol_rel=2e-4,
    relax=0.6,           # under-relaxation for psi update
    psibar_clip_eps=1e-10,

    XPOINT_PREFER = "lower",    #"lower" or "upper"
    XPOINT_GRAD_TOL_FRAC=5e-3,    #sensitivity for grad psi = 0 detection (smaller = stricter)
)

DEFAULT_COILS = [
    ("OB_top", 2.30, +1.10, +2.0e5),
    ("OB_bot", 2.30, -1.10, +2.0e5),
    ("IB_top", 1.20, +0.95, -1.4e5),
    ("IB_bot", 1.20, -0.95, -1.4e5),
    ("VF_top", 2.80, +2.10, +0.6e5),
    ("VF_bot", 2.80, -2.10, +0.6e5),
]


# -------------------------- utils ---------------------------

def next_run_dir(base_dir, prefix):
    os.makedirs(base_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(base_dir, f"{prefix}[0-9][0-9][0-9]")))
    if not existing:
        n = 1
    else:
        m = 0
        for p in existing:
            b = os.path.basename(p)
            mm = re.findall(rf"{prefix}(\d+)", b)
            if mm:
                m = max(m, int(mm[0]))
        n = m + 1

    out = os.path.join(base_dir, f"{prefix}{n:03d}")
    os.makedirs(out, exist_ok=False)
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    return out


def save_plot(path):
    plt.savefig(path, dpi=180)
    plt.close()


def read_params_csv(path, defaults):
    params = dict(defaults)
    if not os.path.isfile(path):
        print(f"[info] PARAMS_CSV not found -> using defaults: {path}")
        return params

    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 2:
                continue
            key = row[0].strip()
            val = row[1].strip()
            if key == "" or key.startswith("#"):
                continue

            if key in ("NR", "NZ", "max_iter"):
                try:
                    params[key] = int(float(val))
                except Exception:
                    pass
            else:
                try:
                    params[key] = float(val)
                except Exception:
                    params[key] = val

    return params


def read_coils_csv(path):
    if not os.path.isfile(path):
        print(f"[info] COILS_CSV not found -> using DEFAULT_COILS: {path}")
        return [dict(name=n, Rc=float(Rc), Zc=float(Zc), I=float(I)) for (n, Rc, Zc, I) in DEFAULT_COILS]

    coils = []
    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                name = (row.get("name", "") or "").strip() or f"coil{len(coils)+1}"
                Rc = float(row["Rc"])
                Zc = float(row["Zc"])
                I  = float(row["I"])
                coils.append(dict(name=name, Rc=Rc, Zc=Zc, I=I))
            except Exception as e:
                print(f"[warn] skipping coil row {row}: {e}")

    if not coils:
        print("[warn] COILS_CSV parsed empty -> using DEFAULT_COILS")
        coils = [dict(name=n, Rc=float(Rc), Zc=float(Zc), I=float(I)) for (n, Rc, Zc, I) in DEFAULT_COILS]
    return coils


# -------------------------- PF coils: psi_ext ---------------------------

def psi_circular_loop(R, Z, Rc, Zc, I=1.0, eps=1e-12):
    """
    ψ from an axisymmetric toroidal filament loop of radius Rc at height Zc.
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    Rp = np.maximum(R, eps)
    dz = Z - Zc
    denom = (Rc + Rp)**2 + dz**2

    k2 = 4.0 * Rc * Rp / denom
    k2 = np.clip(k2, 0.0, 1.0 - 1e-14)
    k = np.sqrt(k2)

    K = ellipk(k2)
    E = ellipe(k2)

    pref = MU0 * I / np.pi
    Aphi = pref * (1.0 / (k + eps)) * np.sqrt(Rc / Rp) * ((1.0 - 0.5*k2) * K - E)
    return Rp * Aphi


def psi_ext_from_coils(R, Z, coils):
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    psi_ext = np.zeros_like(RR, float)
    for c in coils:
        psi_ext += c["I"] * psi_circular_loop(RR, ZZ, c["Rc"], c["Zc"], I=1.0)
    return psi_ext


# -------------------------- GS operator ---------------------------

def build_delta_star_matrix(R, Z):
    """
    Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
    """
    NR = len(R); NZ = len(Z)
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    N = NR * NZ
    A = lil_matrix((N, N), dtype=float)

    def idx(i, j): return i * NZ + j

    for i in range(NR):
        Ri = float(R[i])
        for j in range(NZ):
            k = idx(i, j)
            if i == 0 or i == NR-1 or j == 0 or j == NZ-1:
                continue

            aR = 1.0 / (dR * dR)
            aZ = 1.0 / (dZ * dZ)
            bR = 1.0 / (2.0 * dR)

            # R second derivative
            A[k, idx(i-1, j)] += aR
            A[k, idx(i,   j)] += -2.0 * aR
            A[k, idx(i+1, j)] += aR

            # -(1/R) d/dR
            A[k, idx(i-1, j)] += +(1.0 / Ri) * bR
            A[k, idx(i+1, j)] += -(1.0 / Ri) * bR

            # Z second derivative
            A[k, idx(i, j-1)] += aZ
            A[k, idx(i, j  )] += -2.0 * aZ
            A[k, idx(i, j+1)] += aZ

    return A.tocsr(), dR, dZ


def apply_dirichlet_bc(A, b, psi_bc, R, Z):
    NR = len(R); NZ = len(Z)

    def idx(i, j): return i * NZ + j

    A = A.tolil()
    for i in range(NR):
        for j in range(NZ):
            if i == 0 or i == NR-1 or j == 0 or j == NZ-1:
                k = idx(i, j)
                A.rows[k] = [k]
                A.data[k] = [1.0]
                b[k] = psi_bc[i, j]
    return A.tocsr(), b


# -------------------------- fields & diagnostics ---------------------------

def BR_BZ_from_psi(R, Z, psi):
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    Rcol = R[:, None]
    BR = -(1.0 / Rcol) * dpsi_dZ
    BZ =  (1.0 / Rcol) * dpsi_dR
    return BR, BZ


def Bphi_tf(R, TF_R0, TF_B0):
    Rcol = R[:, None]
    return (TF_B0 * TF_R0) / Rcol


def delta_star_psi(R, Z, psi):
    """Compute Δ*ψ from psi (for jphi diagnostic)."""
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    RR = R[:, None]

    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    d2psi_dR2 = np.gradient(dpsi_dR, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)
    d2psi_dZ2 = np.gradient(dpsi_dZ, dZ, axis=1, edge_order=2)

    return d2psi_dR2 - (1.0 / RR) * dpsi_dR + d2psi_dZ2


def total_Ip(R, Z, jphi, inside_mask):
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    return np.nansum(jphi * inside_mask) * dR * dZ


def find_xpoint_saddle(R, Z, psi, prefer="lower", grad_tol_frac=5e-3):
    """
    Find an X-point candidate as a saddle of psi:
      |∇psi| small AND det(Hessian) < 0.

    Returns:
      (Rx, Zx, psi_x, gbest, detHbest) or None
    """
    R = np.asarray(R); Z = np.asarray(Z)
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])

    # First derivatives
    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)
    gmag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2)

    # Second derivatives / Hessian components
    d2psi_dR2  = np.gradient(dpsi_dR, dR, axis=0, edge_order=2)
    d2psi_dZ2  = np.gradient(dpsi_dZ, dZ, axis=1, edge_order=2)
    d2psi_dRdZ = np.gradient(dpsi_dR, dZ, axis=1, edge_order=2)

    detH = d2psi_dR2 * d2psi_dZ2 - d2psi_dRdZ**2  # saddle -> detH < 0

    # Robust gradient scale (avoid absolute thresholds)
    gscale = np.nanmedian(gmag[2:-2, 2:-2])
    if not np.isfinite(gscale) or gscale <= 0:
        gscale = np.nanmax(gmag)
    if not np.isfinite(gscale) or gscale <= 0:
        return None

    grad_tol = grad_tol_frac * gscale

    # Candidate saddle points in the interior
    cand = (gmag < grad_tol) & (detH < 0)
    cand[:2, :] = False; cand[-2:, :] = False
    cand[:, :2] = False; cand[:, -2:] = False

    if not np.any(cand):
        return None

    # Prefer lower/upper null if requested
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    idxs = np.argwhere(cand)

    if prefer == "lower":
        idxs2 = np.array([ij for ij in idxs if ZZ[ij[0], ij[1]] < 0.0])
        if idxs2.size > 0:
            idxs = idxs2
    elif prefer == "upper":
        idxs2 = np.array([ij for ij in idxs if ZZ[ij[0], ij[1]] > 0.0])
        if idxs2.size > 0:
            idxs = idxs2

    # Choose best candidate: smallest gradient magnitude
    best_i = best_j = None
    best_g = np.inf
    for i, j in idxs:
        gg = gmag[i, j]
        if gg < best_g:
            best_g = gg
            best_i, best_j = int(i), int(j)

    Rx = float(R[best_i])
    Zx = float(Z[best_j])
    psi_x = float(psi[best_i, best_j])
    detHbest = float(detH[best_i, best_j])
    return Rx, Zx, psi_x, float(best_g), detHbest

# -------------------------- profiles p(ψ), F(ψ) ---------------------------

def compute_psibar(psi, psi_axis, psi_sep, eps=1e-12):
    den = psi_sep - psi_axis
    if abs(den) < eps:
        den = eps if den >= 0 else -eps
    return (psi - psi_axis) / den, den


def p_of_psibar(x, p0, alpha_p):
    x = np.clip(x, 0.0, 1.0)
    return p0 * np.power(np.maximum(1.0 - x, 0.0), alpha_p)


def dpdx_of_psibar(x, p0, alpha_p):
    x = np.clip(x, 0.0, 1.0)
    # dp/dx = -p0*alpha*(1-x)^(alpha-1)
    return -p0 * alpha_p * np.power(np.maximum(1.0 - x, 0.0), max(alpha_p - 1.0, 0.0))


def F_of_psibar(x, F0, deltaF, betaF):
    x = np.clip(x, 0.0, 1.0)
    return F0 * (1.0 - deltaF * np.power(x, betaF))


def dFdx_of_psibar(x, F0, deltaF, betaF):
    x = np.clip(x, 0.0, 1.0)
    if betaF == 0:
        return np.zeros_like(x)
    return F0 * (-deltaF) * betaF * np.power(x, betaF - 1.0)


# --------------------------------- main ------------------------------------

def main():
    p = read_params_csv(PARAMS_CSV, DEFAULTS)
    coils = read_coils_csv(COILS_CSV)

    Rmin, Rmax = float(p["Rmin"]), float(p["Rmax"])
    Zmin, Zmax = float(p["Zmin"]), float(p["Zmax"])
    NR, NZ = int(p["NR"]), int(p["NZ"])

    TF_R0, TF_B0 = float(p["TF_R0"]), float(p["TF_B0"])
    R_ref, Z_ref = float(p["R_ref"]), float(p["Z_ref"])

    p0 = float(p["p0"])
    alpha_p = float(p["alpha_p"])
    F0 = float(p["F0"])
    deltaF = float(p["deltaF"])
    betaF = float(p["betaF"])

    max_iter = int(p["max_iter"])
    tol_rel = float(p["tol_rel"])
    relax = float(p["relax"])
    clip_eps = float(p["psibar_clip_eps"])

    XPOINT_PREFER = "lower"
    XPOINT_GRAD_TOL_FRAC = 5e-3#float(["XPOINT_GRAD_TOL_FRAC"])

    # Grid
    R = np.linspace(Rmin, Rmax, NR)
    Z = np.linspace(Zmin, Zmax, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # External BC from PF coils
    psi_ext = psi_ext_from_coils(R, Z, coils)

    # Build operator once
    A, dR, dZ = build_delta_star_matrix(R, Z)

    # Initial guess: vacuum
    psi = psi_ext.copy()

    # Reference point index for psi_sep
    iref = int(np.argmin(np.abs(R - R_ref)))
    jref = int(np.argmin(np.abs(Z - Z_ref)))

    print("GS force-balance (step 3; no X-point)")
    print(f"  PARAMS_CSV: {PARAMS_CSV} ({'found' if os.path.isfile(PARAMS_CSV) else 'missing'})")
    print(f"  COILS_CSV : {COILS_CSV} ({'found' if os.path.isfile(COILS_CSV) else 'missing'})")
    print(f"  grid: {NR} x {NZ}")
    print(f"  profiles: p0={p0:.3e} Pa, alpha_p={alpha_p:g}, F0={F0:.3e} T*m, deltaF={deltaF:g}, betaF={betaF:g}")
    print(f"  psi_sep reference point: (R_ref,Z_ref)=({R_ref:.3f},{Z_ref:.3f})")

    # Picard iteration
    for it in range(1, max_iter + 1):
        psi_axis = float(np.min(psi))

        # --- X-point-based psi_sep with fallback to reference point ---
        xpt = find_xpoint_saddle(R, Z, psi, prefer="lower", grad_tol_frac=5e-3)
        if xpt is None:
            # Fallback: reference point (your old behavior)
            psi_sep = float(psi[iref, jref])
            Rx = Zx = np.nan
            used_xpoint = False
        else:
            Rx, Zx, psi_sep, gbest, detHbest = xpt
            used_xpoint = True

        if it == 1 or it % 5 == 0 or used_xpoint:
            if used_xpoint:
                print(
                    f"[it={it:02d}] psi_sep from X-point "
                    f"at (R={Rx:.3f}, Z={Zx:.3f}), psi_sep={psi_sep:.3e}"
                )
            else:
                print(
                    f"[it={it:02d}] psi_sep from REFERENCE point "
                    f"(R_ref={R[iref]:.3f}, Z_ref={Z[jref]:.3f}), "
                    f"psi_sep={psi_sep:.3e}"
                )

        # Normalized flux (still useful for evaluating profiles p(psibar), F(psibar))
        psibar, den = compute_psibar(psi, psi_axis, psi_sep, eps=clip_eps)

        # Plasma region definition for diverted case:
        # if psi_axis < psi_sep (common convention): closed surfaces have psi <= psi_sep
        if psi_sep > psi_axis:
            inside = (psi <= psi_sep)
        else:
            inside = (psi >= psi_sep)

        inside_mask = inside.astype(np.uint8)

        # Optional: print sometimes
        # print(f"it={it} used_xpoint={used_xpoint} psi_sep={psi_sep:.3e} Rx={Rx} Zx={Zx}")


        # profiles on flux surfaces
        x = np.clip(psibar, 0.0, 1.0)
        p2d = p_of_psibar(x, p0, alpha_p)
        dpdx = dpdx_of_psibar(x, p0, alpha_p)

        F2d = F_of_psibar(x, F0, deltaF, betaF)
        dFdx = dFdx_of_psibar(x, F0, deltaF, betaF)

        # Convert derivatives d/dx to d/dpsi using dx/dpsi = 1/den
        dp_dpsi = dpdx / den
        FFprime = (F2d * dFdx) / den

        # enforce vacuum outside
        dp_dpsi = np.where(inside_mask, dp_dpsi, 0.0)
        FFprime = np.where(inside_mask, FFprime, 0.0)

        # GS RHS
        rhs_2d = -MU0 * (RR**2) * dp_dpsi - FFprime
        rhs = rhs_2d.ravel()

        # Apply Dirichlet BC psi = psi_ext on box boundary
        A2, rhs2 = apply_dirichlet_bc(A, rhs, psi_ext, R, Z)

        # Solve
        psi_new = spsolve(A2, rhs2).reshape((NR, NZ))

        # Under-relax
        psi_upd = (1.0 - relax) * psi + relax * psi_new

        # Convergence measure
        rel = np.linalg.norm((psi_upd - psi).ravel()) / (np.linalg.norm(psi.ravel()) + 1e-30)
        psi = psi_upd

        print(f"  it={it:02d}  rel_change={rel:.3e}  psi_axis={psi_axis:.3e}  psi_sep={psi_sep:.3e}")

        if rel < tol_rel:
            print("  Converged.")
            break

    # Final derived fields
    psi_axis = float(np.min(psi))
    psi_sep = float(psi[iref, jref])
    psibar, den = compute_psibar(psi, psi_axis, psi_sep, eps=clip_eps)
    inside_mask = (psibar <= 1.0).astype(np.uint8)

    x = np.clip(psibar, 0.0, 1.0)
    p2d = p_of_psibar(x, p0, alpha_p)
    F2d = F_of_psibar(x, F0, deltaF, betaF)

    BR, BZ = BR_BZ_from_psi(R, Z, psi)
    Bphi = F2d / (R[:, None])
    # optional: compare against TF model for sanity/plotting
    Bphi_tf_model = Bphi_tf(R, TF_R0, TF_B0)
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2)

    dstar = delta_star_psi(R, Z, psi)
    jphi = -dstar / (MU0 * (R[:, None]))  # diagnostic from GS operator (sign convention)

    Ip = total_Ip(R, Z, jphi, inside_mask)

    # Output dir
    out_dir = next_run_dir(HARD_CODED_BASE, OUT_PREFIX)
    plot_dir = os.path.join(out_dir, "plots")

    # ---------------- Plots ----------------

    # ψ contours
    plt.figure()
    cs = plt.contour(RR, ZZ, psi, levels=50)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
    for c in coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
        plt.text(c["Rc"], c["Zc"], c["name"], fontsize=8)
    plt.axis("equal")
    plt.title("GS force-balance ψ contours (no X-point)")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    plt.grid(True, alpha=0.25)
    save_plot(os.path.join(plot_dir, "psi_contours.png"))

    # inside mask
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, inside_mask, shading="auto")
    plt.colorbar(im, label="inside_mask (1=plasma)")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Plasma region (psibar<=1) with ψ contours")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "inside_mask.png"))

    # pressure map + contours
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, np.where(inside_mask, p2d, np.nan), shading="auto")
    plt.colorbar(im, label="p [Pa]")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Pressure p(ψ) in plasma region")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "pressure.png"))

    # jphi diagnostic
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, np.where(inside_mask, jphi, np.nan), shading="auto")
    plt.colorbar(im, label=r"$j_\phi$ [A/m$^2$] (from -Δ*ψ/μ0R)")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title(r"Toroidal current density diagnostic $j_\phi$")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "jphi.png"))

    # |Bp| streamlines
    Bpol = np.sqrt(BR**2 + BZ**2)
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bpol, shading="auto")
    plt.colorbar(im, label=r"$|B_p|$ [T]")
    plt.streamplot(R, Z, BR.T, BZ.T, density=1.2, linewidth=0.7, arrowsize=1.0)
    plt.axis("equal")
    plt.title("Poloidal field lines over |Bp|")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bp_streamlines.png"))

    # total |B|
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bmag, shading="auto")
    plt.colorbar(im, label=r"$|B|$ [T]")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Total |B| (using Bphi=F/R from GS profiles)")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bmag.png"))

    # 1D midplane check for Bphi profile (F/R) vs TF model
    jmid = int(np.argmin(np.abs(Z - 0.0)))
    plt.figure()
    plt.plot(R, Bphi[:, jmid], label="Bphi from F(ψ)/R")
    plt.plot(R, Bphi_tf_model[:, 0], "--", label="TF model B0 R0 / R")
    plt.title("Midplane Bphi sanity check")
    plt.xlabel("R [m]"); plt.ylabel("Bphi [T]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_plot(os.path.join(plot_dir, "Bphi_midplane_compare.png"))

    #
    plt.figure()
    plt.contour(RR, ZZ, psi, levels=45)
    plt.contour(RR, ZZ, psi, levels=[psi_sep], linewidths=2.0)  # separatrix
    if np.isfinite(Rx):
        plt.plot([Rx], [Zx], "x", markersize=10)
    plt.axis("equal")
    plt.title("ψ contours with separatrix (bold) and X-point (x)")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "separatrix_xpoint.png"))


    # ---------------- Save HDF5 ----------------

    out_h5 = os.path.join(out_dir, H5_NAME)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("R", data=R)
        f.create_dataset("Z", data=Z)
        f.create_dataset("psi", data=psi)
        f.create_dataset("psi_ext", data=psi_ext)
        f.create_dataset("psibar", data=psibar)
        f.create_dataset("inside_mask", data=inside_mask)

        f.create_dataset("p", data=p2d)
        f.create_dataset("F", data=F2d)

        # store source terms used
        dpdx = dpdx_of_psibar(np.clip(psibar, 0.0, 1.0), p0, alpha_p)
        dFdx = dFdx_of_psibar(np.clip(psibar, 0.0, 1.0), F0, deltaF, betaF)
        dp_dpsi = np.where(inside_mask, dpdx / den, 0.0)
        FFprime = np.where(inside_mask, (F2d * dFdx) / den, 0.0)

        f.create_dataset("dp_dpsi", data=dp_dpsi)
        f.create_dataset("FFprime", data=FFprime)

        f.create_dataset("BR", data=BR)
        f.create_dataset("BZ", data=BZ)
        f.create_dataset("Bphi", data=Bphi)
        f.create_dataset("Bmag", data=Bmag)

        grp = f.create_group("gs_diagnostics")
        grp.create_dataset("delta_star_psi", data=dstar)
        grp.create_dataset("jphi", data=jphi)
        grp.attrs["Ip_from_jphi_inside_mask"] = float(Ip)

        g = f.create_group("pf_coils")
        g.attrs["ncoils"] = len(coils)
        g.create_dataset("name", data=np.array([c["name"].encode("utf8") for c in coils]))
        g.create_dataset("Rc", data=np.array([c["Rc"] for c in coils], float))
        g.create_dataset("Zc", data=np.array([c["Zc"] for c in coils], float))
        g.create_dataset("I",  data=np.array([c["I"]  for c in coils], float))

        f.attrs["PARAMS_CSV"] = PARAMS_CSV
        f.attrs["COILS_CSV"] = COILS_CSV

        f.attrs["Rmin"] = float(Rmin); f.attrs["Rmax"] = float(Rmax)
        f.attrs["Zmin"] = float(Zmin); f.attrs["Zmax"] = float(Zmax)
        f.attrs["NR"] = int(NR); f.attrs["NZ"] = int(NZ)
        f.attrs["dR"] = float(dR); f.attrs["dZ"] = float(dZ)

        f.attrs["TF_R0"] = float(TF_R0)
        f.attrs["TF_B0"] = float(TF_B0)

        f.attrs["R_ref"] = float(R_ref)
        f.attrs["Z_ref"] = float(Z_ref)
        f.attrs["psi_axis"] = float(psi_axis)
        f.attrs["psi_sep_refpoint"] = float(psi_sep)

        f.attrs["profile_p0"] = float(p0)
        f.attrs["profile_alpha_p"] = float(alpha_p)
        f.attrs["profile_F0"] = float(F0)
        f.attrs["profile_deltaF"] = float(deltaF)
        f.attrs["profile_betaF"] = float(betaF)

        f.attrs["max_iter"] = int(max_iter)
        f.attrs["tol_rel"] = float(tol_rel)
        f.attrs["relax"] = float(relax)

        f.attrs["used_xpoint_for_psi_sep"] = int(used_xpoint)
        f.attrs["xpoint_R"] = float(Rx) if np.isfinite(Rx) else np.nan
        f.attrs["xpoint_Z"] = float(Zx) if np.isfinite(Zx) else np.nan
        f.attrs["psi_sep"] = float(psi_sep)

        f.attrs["notes"] = (
            "Free-boundary GS force-balance solve (no X-point): Δ*ψ = -μ0 R^2 p'(ψ) - F F'(ψ). "
            "Edge ψ_sep defined by reference point (R_ref,Z_ref), plasma region by psibar<=1. "
            "PF coils provide ψ_ext Dirichlet boundary on the outer box."
        )

    print(f"Saved -> {out_dir}")
    print(f"  H5   : {out_h5}")
    print(f"  Plots: {plot_dir}")
    print(f"  Diagnostic Ip (from jphi=-Δ*ψ/(μ0R) inside mask): {Ip:.3e} A")


if __name__ == "__main__":
    main()
