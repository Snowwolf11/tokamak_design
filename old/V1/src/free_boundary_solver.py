#!/usr/bin/env python3
"""
Free-boundary GS (Step 3): vacuum + plasma with jphi = jphi(psi)
===============================================================

We solve (Picard iteration):
    Δ* psi = - μ0 R jphi(psi)

- Vacuum region: jphi = 0 where psibar > 1
- Plasma region defined self-consistently by psibar in [0,1]
- External PF coils: provide Dirichlet boundary condition psi = psi_ext on the outer box boundary
- jphi profile is a flux-function:
      jphi(psi) = j0 * (1 - psibar)^alpha_j,  for psibar in [0,1]
  and we rescale j0 each iteration so that total plasma current matches Ip_target.

This is a tidy, robust "machine -> equilibrium" step that stays GS-compatible.
It is not yet full MHD force balance (that would use p'(psi) and FF'(psi)),
but it is a correct step-3 free-boundary equilibrium workflow.

Outputs:
  HARD_CODED_BASE/gs_free_boundaryXXX/
      equilibrium.h5
      plots/*.png

CSV inputs (hardcoded paths):
  - PARAMS_CSV: key,value pairs
  - COILS_CSV : columns name,Rc,Zc,I (PF coils)

Missing CSV or missing keys -> defaults are used.
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
HARD_CODED_BASE = "/Users/leon/Desktop/python_skripte/tokamak_design/old/runs"

PARAMS_CSV = "/Users/leon/Desktop/python_skripte/tokamak_design/old/inputs/gs_free_boundary_params.csv"
COILS_CSV  = "/Users/leon/Desktop/python_skripte/tokamak_design/old/inputs/pf_coils.csv"

OUT_PREFIX = "gs_free_boundary"
H5_NAME = "equilibrium.h5"

# -------------------------- defaults ---------------------------

DEFAULTS = dict(
    # domain grid
    Rmin=0.7, Rmax=3.4, Zmin=-2.6, Zmax=2.6,
    NR=257, NZ=257,

    # TF model (stored + used for Bmag plots, doesn't enter psi)
    TF_R0=1.65,
    TF_B0=2.5,

    # reference point to define psi_b/psi_lcfs guess (used only for initial psi_sep)
    R_ref=2.2, Z_ref=0.0,

    # plasma current target and profile
    Ip_target=0.8e6,      # [A]
    alpha_j=1.5,          # jphi ~ (1-psibar)^alpha_j
    psibar_clip_eps=1e-8, # numerical

    # iteration controls
    max_iter=40,
    tol_rel=2e-4,
    relax=0.6,            # under-relax psi update
)

DEFAULT_COILS = [
    # name, Rc [m], Zc [m], I [A]
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
            # try float/int
            if key in ("NR", "NZ", "max_iter"):
                try:
                    params[key] = int(float(val))
                except Exception:
                    pass
            else:
                try:
                    params[key] = float(val)
                except Exception:
                    # allow strings if ever needed
                    params[key] = val

    return params


def read_coils_csv(path):
    if not os.path.isfile(path):
        print(f"[info] COILS_CSV not found -> using DEFAULT_COILS: {path}")
        coils = []
        for name, Rc, Zc, I in DEFAULT_COILS:
            coils.append(dict(name=name, Rc=float(Rc), Zc=float(Zc), I=float(I)))
        return coils

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
        for name, Rc, Zc, I in DEFAULT_COILS:
            coils.append(dict(name=name, Rc=float(Rc), Zc=float(Zc), I=float(I)))
    return coils


# -------------------------- PF coils: psi_ext ---------------------------

def psi_circular_loop(R, Z, Rc, Zc, I=1.0, eps=1e-12):
    """
    ψ from a toroidal filament loop of radius Rc at height Zc (axisymmetric ring).
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


# -------------------------- fields ---------------------------

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


# -------------------------- free-boundary iteration ---------------------------

def compute_psibar(psi, psi_axis, psi_sep, eps=1e-12):
    den = (psi_sep - psi_axis)
    if abs(den) < eps:
        den = np.sign(den) * eps if den != 0 else eps
    psibar = (psi - psi_axis) / den
    return psibar


def jphi_profile_from_psibar(psibar, j0, alpha_j, clip_eps=1e-12):
    x = np.clip(psibar, 0.0, 1.0)
    return j0 * np.power(np.maximum(1.0 - x, 0.0), alpha_j)


def total_Ip(R, Z, jphi, inside_mask):
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    dR = float(R[1] - R[0]); dZ = float(Z[1] - Z[0])
    return np.nansum(jphi * inside_mask) * dR * dZ


def main():
    # ------------------- read inputs -------------------
    p = read_params_csv(PARAMS_CSV, DEFAULTS)
    coils = read_coils_csv(COILS_CSV)

    Rmin, Rmax = float(p["Rmin"]), float(p["Rmax"])
    Zmin, Zmax = float(p["Zmin"]), float(p["Zmax"])
    NR, NZ = int(p["NR"]), int(p["NZ"])

    TF_R0, TF_B0 = float(p["TF_R0"]), float(p["TF_B0"])
    Ip_target = float(p["Ip_target"])
    alpha_j = float(p["alpha_j"])
    relax = float(p["relax"])
    max_iter = int(p["max_iter"])
    tol_rel = float(p["tol_rel"])

    R_ref, Z_ref = float(p["R_ref"]), float(p["Z_ref"])
    clip_eps = float(p["psibar_clip_eps"])

    # ------------------- grid -------------------
    R = np.linspace(Rmin, Rmax, NR)
    Z = np.linspace(Zmin, Zmax, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # ------------------- external vacuum flux for BC -------------------
    psi_ext = psi_ext_from_coils(R, Z, coils)

    # ------------------- build operator once -------------------
    A, dR, dZ = build_delta_star_matrix(R, Z)

    # ------------------- initial guess -------------------
    psi = psi_ext.copy()

    # initial axis and psi_sep guess
    psi_axis = float(np.min(psi))
    # psi_sep guess from reference point
    iref = int(np.argmin(np.abs(R - R_ref)))
    jref = int(np.argmin(np.abs(Z - Z_ref)))
    psi_sep = float(psi[iref, jref])

    # initial j0 guess: scale so Ip roughly matches for an initial inside region
    psibar = compute_psibar(psi, psi_axis, psi_sep)
    inside = (psibar <= 1.0)
    j0 = 1e6  # rough initial
    jphi = jphi_profile_from_psibar(psibar, j0, alpha_j, clip_eps=clip_eps)
    Ip0 = total_Ip(R, Z, jphi, inside)
    if abs(Ip0) > 1e-12:
        j0 *= (Ip_target / Ip0)

    # ------------------- Picard iteration -------------------
    print("Free-boundary GS (step 3)")
    print(f"  params csv: {PARAMS_CSV} ({'found' if os.path.isfile(PARAMS_CSV) else 'missing'})")
    print(f"  coils  csv: {COILS_CSV} ({'found' if os.path.isfile(COILS_CSV) else 'missing'})")
    print(f"  NRxNZ = {NR} x {NZ}")
    print(f"  Target Ip = {Ip_target:.3e} A")

    for it in range(1, max_iter + 1):
        # axis is minimum psi (typical for tokamak convention)
        psi_axis_new = float(np.min(psi))

        # keep psi_sep fixed to reference point value (simple step-3 choice)
        psi_sep_new = float(psi[iref, jref])

        psibar = compute_psibar(psi, psi_axis_new, psi_sep_new)
        inside = (psibar <= 1.0)

        # current profile on flux surfaces
        jphi_shape = jphi_profile_from_psibar(psibar, 1.0, alpha_j, clip_eps=clip_eps)
        Ip_shape = total_Ip(R, Z, jphi_shape, inside)

        if abs(Ip_shape) < 1e-20:
            print("[warn] Plasma region vanished (Ip_shape ~ 0). Adjust R_ref/Z_ref or coil set.")
            break

        j0 = Ip_target / Ip_shape
        jphi = j0 * jphi_shape * inside  # force vacuum region to zero

        # solve linear system: Δ*psi = -μ0 R jphi  with BC from psi_ext
        rhs = (-MU0 * RR * jphi).ravel()
        A2, rhs2 = apply_dirichlet_bc(A, rhs, psi_ext, R, Z)
        psi_new = spsolve(A2, rhs2).reshape((NR, NZ))

        # under-relax update
        psi_upd = (1.0 - relax) * psi + relax * psi_new

        # convergence metric
        num = np.linalg.norm((psi_upd - psi).ravel())
        den = np.linalg.norm(psi.ravel()) + 1e-30
        rel = num / den

        psi = psi_upd
        psi_axis, psi_sep = psi_axis_new, psi_sep_new

        Ip_now = total_Ip(R, Z, jphi, inside)

        print(f"  it={it:02d}  rel_change={rel:.3e}  psi_axis={psi_axis:.3e}  psi_sep={psi_sep:.3e}  j0={j0:.3e}  Ip={Ip_now:.3e}")

        if rel < tol_rel:
            print("  Converged.")
            break

    # final masks and fields
    psibar = compute_psibar(psi, psi_axis, psi_sep)
    inside_mask = (psibar <= 1.0).astype(np.uint8)

    BR, BZ = BR_BZ_from_psi(R, Z, psi)
    Bphi = Bphi_tf(R, TF_R0, TF_B0)
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2)

    # ------------------- output dir -------------------
    out_dir = next_run_dir(HARD_CODED_BASE, OUT_PREFIX)
    plot_dir = os.path.join(out_dir, "plots")

    # ------------------- plots -------------------
    # 1) psi_total contours + coils
    plt.figure()
    cs = plt.contour(RR, ZZ, psi, levels=50)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
    for c in coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
        plt.text(c["Rc"], c["Zc"], c["name"], fontsize=8)
    plt.axis("equal")
    plt.title("Free-boundary ψ (plasma jφ= jφ(ψ) + PF vacuum BC)")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    plt.grid(True, alpha=0.25)
    save_plot(os.path.join(plot_dir, "psi_total_contours.png"))

    # 2) inside mask (LCFS proxy)
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, inside_mask, shading="auto")
    plt.colorbar(im, label="inside_mask (1=plasma)")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Plasma region defined by psibar<=1 (LCFS proxy)")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "inside_mask.png"))

    # 3) jphi map
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, jphi, shading="auto")
    plt.colorbar(im, label=r"$j_\phi$ [A/m$^2$]")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title(r"Toroidal plasma current density $j_\phi(\psi)$")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "jphi.png"))

    # 4) |Bp| + streamlines
    Bpol = np.sqrt(BR**2 + BZ**2)
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bpol, shading="auto")
    plt.colorbar(im, label=r"$|B_p|$ [T]")
    plt.streamplot(R, Z, BR.T, BZ.T, density=1.2, linewidth=0.7, arrowsize=1.0)
    plt.axis("equal")
    plt.title("Poloidal field lines over |Bp|")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bp_streamlines.png"))

    # 5) total |B| (includes TF)
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bmag, shading="auto")
    plt.colorbar(im, label=r"$|B|$ [T]")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Total |B| including TF model")
    plt.xlabel("R [m]"); plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bmag.png"))

    # ------------------- save HDF5 -------------------
    out_h5 = os.path.join(out_dir, H5_NAME)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("R", data=R)
        f.create_dataset("Z", data=Z)
        f.create_dataset("psi", data=psi)
        f.create_dataset("psi_ext", data=psi_ext)
        f.create_dataset("psibar", data=psibar)
        f.create_dataset("inside_mask", data=inside_mask)

        f.create_dataset("jphi", data=jphi)
        f.create_dataset("BR", data=BR)
        f.create_dataset("BZ", data=BZ)
        f.create_dataset("Bphi", data=Bphi)
        f.create_dataset("Bmag", data=Bmag)

        g = f.create_group("pf_coils")
        g.attrs["ncoils"] = len(coils)
        g.create_dataset("name", data=np.array([c["name"].encode("utf8") for c in coils]))
        g.create_dataset("Rc", data=np.array([c["Rc"] for c in coils], float))
        g.create_dataset("Zc", data=np.array([c["Zc"] for c in coils], float))
        g.create_dataset("I",  data=np.array([c["I"]  for c in coils], float))

        # scalar attributes
        f.attrs["PARAMS_CSV"] = PARAMS_CSV
        f.attrs["COILS_CSV"] = COILS_CSV
        f.attrs["Rmin"] = float(Rmin); f.attrs["Rmax"] = float(Rmax)
        f.attrs["Zmin"] = float(Zmin); f.attrs["Zmax"] = float(Zmax)
        f.attrs["NR"] = int(NR); f.attrs["NZ"] = int(NZ)
        f.attrs["dR"] = float(dR); f.attrs["dZ"] = float(dZ)

        f.attrs["TF_R0"] = float(TF_R0)
        f.attrs["TF_B0"] = float(TF_B0)

        f.attrs["Ip_target"] = float(Ip_target)
        f.attrs["alpha_j"] = float(alpha_j)
        f.attrs["relax"] = float(relax)
        f.attrs["max_iter"] = int(max_iter)
        f.attrs["tol_rel"] = float(tol_rel)

        f.attrs["R_ref"] = float(R_ref)
        f.attrs["Z_ref"] = float(Z_ref)
        f.attrs["psi_axis"] = float(psi_axis)
        f.attrs["psi_sep_refpoint"] = float(psi_sep)

        f.attrs["notes"] = (
            "Step-3 free-boundary solve with vacuum region and flux-function toroidal current jphi(psi). "
            "Boundary condition psi=psi_ext from PF coils on outer box. "
            "Plasma region defined by psibar<=1 with psi_sep taken from a reference point (R_ref,Z_ref). "
            "Next step for diverted equilibria: allow asymmetric PF currents and solve for a separatrix/X-point."
        )

    print(f"Saved -> {out_dir}")
    print(f"  H5   : {out_h5}")
    print(f"  Plots: {plot_dir}")


if __name__ == "__main__":
    main()
