#!/usr/bin/env python3
"""
Equilibrium-like solve from: prescribed plasma jphi(R,Z) + PF coils + TF field
=============================================================================

We solve for poloidal flux psi on a rectangular (R,Z) grid:
    Δ* psi = -μ0 R jphi_plasma(R,Z)

Boundary condition:
    psi = psi_ext (vacuum poloidal flux from PF coils) on the outer box boundary.

Then compute:
    BR = -(1/R) ∂psi/∂Z
    BZ =  (1/R) ∂psi/∂R
    Bphi = B0*R0/R   (simple TF model, not from psi)
    Bmag = sqrt(BR^2 + BZ^2 + Bphi^2)

We also compute psi_ext and store it.

Important:
- This is a GREAT "machine->field->flux surfaces" forward model.
- It is NOT yet a fully consistent MHD equilibrium unless jphi is a flux function
  (or comes from p'(psi), FF'(psi)). That’s the next upgrade after this step.
"""

import os
import re
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.special import ellipk, ellipe
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

MU0 = 4e-7 * np.pi

# -------------------------- hardcoded output root ---------------------------
HARD_CODED_BASE = "/Users/leon/Desktop/python_skripte/tokamak_design/runs"
HDF5_NAME = "equilibrium.h5"


# ----------------------------- directory helper ----------------------------

def next_design_dir(base_dir, prefix="equilibrium_design"):
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

    out_dir = os.path.join(base_dir, f"{prefix}{n:03d}")
    os.makedirs(out_dir, exist_ok=False)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    return out_dir


def save_plot(path):
    plt.savefig(path, dpi=180)
    plt.close()


# -------------------------- PF coil vacuum psi -----------------------------

def psi_circular_loop(R, Z, Rc, Zc, I=1.0, eps=1e-12):
    """
    Poloidal flux ψ(R,Z) from a circular toroidal filament loop centered at (Rc,Zc) with current I.
    Standard elliptic-integral expression via Aphi, then ψ = R Aphi.
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


def psi_ext_from_pf_coils(R, Z, coils):
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    psi_ext = np.zeros_like(RR, float)

    # also store basis for later fitting (unit current)
    basis = np.zeros((len(coils), RR.shape[0], RR.shape[1]), float)

    for i, c in enumerate(coils):
        Rc = float(c["Rc"])
        Zc = float(c["Zc"])
        I = float(c["I"])
        psi_unit = psi_circular_loop(RR, ZZ, Rc, Zc, I=1.0)
        basis[i] = psi_unit
        psi_ext += I * psi_unit

    return psi_ext, basis


# -------------------------- prescribed plasma jphi -------------------------

def jphi_gaussian(RR, ZZ, R0, Z0, sigR, sigZ, j0):
    """Smooth toroidal plasma current density profile jphi(R,Z) [A/m^2]."""
    return j0 * np.exp(-((RR - R0)**2)/(sigR**2) - ((ZZ - Z0)**2)/(sigZ**2))


# -------------------------- GS operator Δ* discretization ------------------

def build_delta_star_matrix(R, Z):
    """
    Build sparse matrix for Δ* acting on psi on an (NR,NZ) grid with indexing (i,j)=(R,Z).
    Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²

    Dirichlet boundaries are handled outside (we set rows to identity).
    """
    NR = len(R)
    NZ = len(Z)
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    N = NR * NZ
    A = lil_matrix((N, N), dtype=float)

    def idx(i, j):
        return i * NZ + j

    for i in range(NR):
        Ri = float(R[i])
        for j in range(NZ):
            k = idx(i, j)

            # boundary nodes: fill later as Dirichlet identity
            if i == 0 or i == NR-1 or j == 0 or j == NZ-1:
                continue

            # central differences
            aR = 1.0 / (dR * dR)
            aZ = 1.0 / (dZ * dZ)
            bR = 1.0 / (2.0 * dR)

            # ∂²/∂R² term
            A[k, idx(i-1, j)] += aR
            A[k, idx(i,   j)] += -2.0 * aR
            A[k, idx(i+1, j)] += aR

            # -(1/R)∂/∂R term
            A[k, idx(i-1, j)] += +(1.0 / Ri) * bR
            A[k, idx(i+1, j)] += -(1.0 / Ri) * bR

            # ∂²/∂Z² term
            A[k, idx(i, j-1)] += aZ
            A[k, idx(i, j  )] += -2.0 * aZ
            A[k, idx(i, j+1)] += aZ

    return A.tocsr(), dR, dZ


def apply_dirichlet(A, b, psi_bc, R, Z):
    """Apply Dirichlet BC psi=psi_bc on all outer boundary nodes."""
    NR = len(R)
    NZ = len(Z)

    def idx(i, j):
        return i * NZ + j

    A = A.tolil()
    for i in range(NR):
        for j in range(NZ):
            if i == 0 or i == NR-1 or j == 0 or j == NZ-1:
                k = idx(i, j)
                A.rows[k] = [k]
                A.data[k] = [1.0]
                b[k] = psi_bc[i, j]
    return A.tocsr(), b


# -------------------------- B from psi + TF model --------------------------

def BR_BZ_from_psi(R, Z, psi):
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    Rcol = R[:, None]
    BR = -(1.0 / Rcol) * dpsi_dZ
    BZ =  (1.0 / Rcol) * dpsi_dR
    return BR, BZ


def Bphi_tf(R, R0, B0):
    """Simple TF: Bphi(R) = B0 * R0 / R."""
    Rcol = R[:, None]
    return (B0 * R0) / Rcol


# --------------------------------- main ------------------------------------

def main():
    # ---------------------------------------------------------------------
    # 1) Machine-level inputs (edit these)
    # ---------------------------------------------------------------------

    # Desired “device scale” for TF field model
    R0_tf = 1.65   # [m]
    B0_tf = 2.5    # [T] at R0_tf

    # PF coils (shape/position control) — toroidal filaments in axisymmetric model, Loop: [ Rc*cos(phi), Rc*sin(phi), zc]
    pf_coils = [
        {"name": "OB_top", "Rc": 2.30, "Zc": +1.10, "I": +2.0e5},
        {"name": "OB_bot", "Rc": 2.30, "Zc": -1.10, "I": +2.0e5},
        {"name": "IB_top", "Rc": 1.20, "Zc": +0.95, "I": -1.4e5},
        {"name": "IB_bot", "Rc": 1.20, "Zc": -0.95, "I": -1.4e5},
        {"name": "VF_top", "Rc": 2.80, "Zc": +2.10, "I": +0.6e5},
        {"name": "VF_bot", "Rc": 2.80, "Zc": -2.10, "I": +0.6e5
        },
    ]

    # Plasma toroidal current density model jphi(R,Z)
    # This is your “smooth plasma current profile in the poloidal plane”.
    plasma = dict(
        R_center=1.65,  # [m]
        Z_center=0.0,   # [m]
        sigR=0.35,      # [m]
        sigZ=0.55,      # [m]
        j0=3.0e6,       # [A/m^2] amplitude (tune later)
    )

    # Computational box (must include PF coils and plasma region)
    Rmin, Rmax = 0.7, 3.4
    Zmin, Zmax = -2.6, 2.6
    NR, NZ = 321, 321

    # ---------------------------------------------------------------------
    # 2) Grid
    # ---------------------------------------------------------------------
    R = np.linspace(Rmin, Rmax, NR)
    Z = np.linspace(Zmin, Zmax, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # ---------------------------------------------------------------------
    # 3) External PF-coil vacuum psi on this grid (for BC and diagnostics)
    # ---------------------------------------------------------------------
    psi_ext, psi_basis = psi_ext_from_pf_coils(R, Z, pf_coils)

    # ---------------------------------------------------------------------
    # 4) Prescribed plasma jphi(R,Z)
    # ---------------------------------------------------------------------
    jphi_plasma = jphi_gaussian(
        RR, ZZ,
        R0=plasma["R_center"],
        Z0=plasma["Z_center"],
        sigR=plasma["sigR"],
        sigZ=plasma["sigZ"],
        j0=plasma["j0"]
    )

    # ---------------------------------------------------------------------
    # 5) Solve Δ*psi = -μ0 R jphi_plasma with Dirichlet BC psi=psi_ext on boundary
    # ---------------------------------------------------------------------
    A, dR, dZ = build_delta_star_matrix(R, Z)

    rhs = (-MU0 * (RR) * jphi_plasma).ravel()
    A2, rhs2 = apply_dirichlet(A, rhs, psi_ext, R, Z)

    psi = spsolve(A2, rhs2).reshape((NR, NZ))

    # ---------------------------------------------------------------------
    # 6) Fields: BR,BZ from psi; Bphi from TF model
    # ---------------------------------------------------------------------
    BR, BZ = BR_BZ_from_psi(R, Z, psi)
    Bphi = Bphi_tf(R, R0_tf, B0_tf)
    Bmag = np.sqrt(BR**2 + BZ**2 + Bphi**2)

    # ---------------------------------------------------------------------
    # 7) Output directory + plots
    # ---------------------------------------------------------------------
    out_dir = next_design_dir(HARD_CODED_BASE, prefix="equilibrium_design")
    plot_dir = os.path.join(out_dir, "plots")

    # Plot 1: psi contours (total) with PF coil markers
    plt.figure()
    cs = plt.contour(RR, ZZ, psi, levels=45)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
    for c in pf_coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
        plt.text(c["Rc"], c["Zc"], c["name"], fontsize=8)
    plt.axis("equal")
    plt.title("Total poloidal flux ψ (PF coils as BC + prescribed plasma jφ source)")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.grid(True, alpha=0.25)
    save_plot(os.path.join(plot_dir, "psi_total_contours.png"))

    # Plot 2: psi_ext contours (external vacuum) for reference
    plt.figure()
    cs = plt.contour(RR, ZZ, psi_ext, levels=45)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
    for c in pf_coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
    plt.axis("equal")
    plt.title("External (PF-coil) vacuum flux ψ_ext")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.grid(True, alpha=0.25)
    save_plot(os.path.join(plot_dir, "psi_ext_contours.png"))

    # Plot 3: prescribed jphi
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, jphi_plasma, shading="auto")
    plt.colorbar(im, label=r"$j_\phi$ [A/m$^2$]")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Prescribed plasma toroidal current density with ψ contours")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "jphi_prescribed.png"))

    # Plot 4: |Bp| with streamlines
    Bpol = np.sqrt(BR**2 + BZ**2)
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bpol, shading="auto")
    plt.colorbar(im, label=r"$|B_p|$ [T]")
    plt.streamplot(R, Z, BR.T, BZ.T, density=1.2, linewidth=0.7, arrowsize=1.0)
    plt.axis("equal")
    plt.title("Poloidal field lines (from ψ) over |Bp|")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bp_streamlines.png"))

    # Plot 5: total |B| (includes TF)
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bmag, shading="auto")
    plt.colorbar(im, label=r"$|B|$ [T]")
    plt.contour(RR, ZZ, psi, levels=25, linewidths=0.8)
    plt.axis("equal")
    plt.title("Total |B| including TF model (Bφ = B0 R0 / R)")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bmag_total.png"))

    # ---------------------------------------------------------------------
    # 8) Save HDF5 (same “style” as your pipeline: self-contained + attrs)
    # ---------------------------------------------------------------------
    out_h5 = os.path.join(out_dir, HDF5_NAME)
    with h5py.File(out_h5, "w") as f:
        # Raw grid
        f.create_dataset("R", data=R)
        f.create_dataset("Z", data=Z)

        # Fluxes
        f.create_dataset("psi", data=psi)
        f.create_dataset("psi_ext", data=psi_ext)

        # Fields
        f.create_dataset("BR", data=BR)
        f.create_dataset("BZ", data=BZ)
        f.create_dataset("Bphi", data=Bphi)
        f.create_dataset("Bmag", data=Bmag)

        # Sources / models
        f.create_dataset("jphi_plasma", data=jphi_plasma)
        f.create_dataset("psi_basis_unit_current", data=psi_basis)  # useful for later fitting

        # Coil metadata
        g = f.create_group("pf_coils")
        g.attrs["ncoils"] = len(pf_coils)
        g.create_dataset("name", data=np.array([c["name"].encode("utf8") for c in pf_coils]))
        g.create_dataset("Rc", data=np.array([c["Rc"] for c in pf_coils], float))
        g.create_dataset("Zc", data=np.array([c["Zc"] for c in pf_coils], float))
        g.create_dataset("I",  data=np.array([c["I"]  for c in pf_coils], float))

        # Attributes
        f.attrs["Rmin"] = float(Rmin); f.attrs["Rmax"] = float(Rmax)
        f.attrs["Zmin"] = float(Zmin); f.attrs["Zmax"] = float(Zmax)
        f.attrs["NR"] = int(NR); f.attrs["NZ"] = int(NZ)
        f.attrs["dR"] = float(dR); f.attrs["dZ"] = float(dZ)

        f.attrs["TF_R0"] = float(R0_tf)
        f.attrs["TF_B0"] = float(B0_tf)

        for k, v in plasma.items():
            f.attrs[f"plasma_{k}"] = float(v)

        f.attrs["notes"] = (
            "Solved Δ*psi = -μ0 R jphi_plasma with Dirichlet BC psi=psi_ext from PF coils on domain boundary. "
            "This is a machine-driven axisymmetric field model. Not yet a full MHD equilibrium unless sources are flux functions."
        )

    print(f"Equilibrium-like outputs -> {out_dir}")
    print(f"  Data : {out_h5}")
    print(f"  Plots: {plot_dir}")


if __name__ == "__main__":
    main()
