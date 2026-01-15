#!/usr/bin/env python3
"""
Vacuum field from prescribed PF coils (no fitting)
==================================================

Input: user-specified coil list (positions + currents).
Output:
  - Vacuum poloidal flux psi_vac(R,Z)
  - Vacuum poloidal field BR, BZ from psi
  - Flux surfaces (psi contours) and sensible B-field plots
  - Saved to: HARD_CODED_BASE/vacuum_designXXX/
      - vacuum.h5
      - plots/*.png

Coil model:
  - Axisymmetric toroidal filament (circular loop) at (Rc, Zc) with current I
  - Uses elliptic integrals (SciPy).

Notes:
  - This computes the *poloidal* field from PF coils.
  - It does NOT include plasma current (no GS), and does not include TF coils' poloidal contribution.
"""

import os
import re
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi


# -------------------------- hardcoded output root ---------------------------
HARD_CODED_BASE = "/Users/leon/Desktop/python_skripte/tokamak_design/runs"
HDF5_NAME = "vacuum.h5"


# ----------------------------- directory helper ----------------------------

def next_design_dir(base_dir, prefix="vacuum_design"):
    """
    Creates base_dir/prefixXXX with XXX counting up from 001.
    Returns the new directory path.
    """
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


# -------------------------- coil -> psi (elliptic) --------------------------

def psi_circular_loop(R, Z, Rc, Zc, I=1.0, eps=1e-12):
    """
    Poloidal flux ψ(R,Z) from a circular filament loop (toroidal current),
    centered at (Rc, Zc), carrying current I.

    Uses a standard vector potential Aphi expression:
      k^2 = 4 Rc R / ((Rc + R)^2 + (Z-Zc)^2)
      Aphi = μ0 I / (π k) * sqrt(Rc/R) * [ (1 - k^2/2) K(k^2) - E(k^2) ]
      ψ = R Aphi

    Returns ψ same shape as R,Z.
    """
    R = np.asarray(R, float)
    Z = np.asarray(Z, float)

    Rp = np.maximum(R, eps)
    dz = Z - Zc
    denom = (Rc + Rp)**2 + dz**2

    k2 = 4.0 * Rc * Rp / denom
    k2 = np.clip(k2, 0.0, 1.0 - 1e-14)  # stability
    k = np.sqrt(k2)

    K = ellipk(k2)
    E = ellipe(k2)

    pref = MU0 * I / np.pi
    Aphi = pref * (1.0 / (k + eps)) * np.sqrt(Rc / Rp) * ((1.0 - 0.5*k2) * K - E)

    return Rp * Aphi


def vacuum_psi_from_coils(R, Z, coils):
    """
    coils: list of dicts with keys:
        - name (str)
        - shape (str): currently only "circular_filament"
        - Rc (m)
        - Zc (m)
        - I (A)

    Returns:
      psi (NR,NZ), plus per-coil contributions basis (ncoils, NR, NZ) with unit current.
    """
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    n = len(coils)

    psi = np.zeros_like(RR, dtype=float)
    basis = np.zeros((n, RR.shape[0], RR.shape[1]), dtype=float)

    for i, c in enumerate(coils):
        shape = c.get("shape", "circular_filament")
        if shape != "circular_filament":
            raise ValueError(f"Unsupported coil shape '{shape}'. Only 'circular_filament' is implemented.")

        Rc = float(c["Rc"])
        Zc = float(c["Zc"])
        I = float(c["I"])

        psi_unit = psi_circular_loop(RR, ZZ, Rc, Zc, I=1.0)
        basis[i] = psi_unit
        psi += I * psi_unit

    return psi, basis


def BR_BZ_from_psi(R, Z, psi):
    """
    Compute BR, BZ from psi using:
      BR = -(1/R) dψ/dZ
      BZ =  (1/R) dψ/dR
    """
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])

    dpsi_dR = np.gradient(psi, dR, axis=0, edge_order=2)
    dpsi_dZ = np.gradient(psi, dZ, axis=1, edge_order=2)

    Rcol = R[:, None]
    BR = -(1.0 / Rcol) * dpsi_dZ
    BZ =  (1.0 / Rcol) * dpsi_dR
    return BR, BZ


# --------------------------------- main ------------------------------------

def main():
    # ---------------------------
    # 1) Define coils (INPUT)
    # ---------------------------
    # This is where you control "coil shape, number, and current".
    # Right now: shape="circular_filament" at (Rc,Zc) with toroidal current I.
    coils = [
        {"name": "OB_top", "shape": "circular_filament", "Rc": 2.30, "Zc": +1.10, "I": +2.0e5},
        {"name": "OB_bot", "shape": "circular_filament", "Rc": 2.30, "Zc": -1.10, "I": +2.0e5},
        {"name": "IB_top", "shape": "circular_filament", "Rc": 1.20, "Zc": +0.95, "I": -1.4e5},
        {"name": "IB_bot", "shape": "circular_filament", "Rc": 1.20, "Zc": -0.95, "I": -1.4e5},
        {"name": "VF_top", "shape": "circular_filament", "Rc": 2.80, "Zc": +2.10, "I": +0.6e5},
        {"name": "VF_bot", "shape": "circular_filament", "Rc": 2.80, "Zc": -2.10, "I": +0.6e5},
    ]

    # ---------------------------
    # 2) Define grid (INPUT)
    # ---------------------------
    # Choose a box that covers coils + region you care about.
    Rmin, Rmax = 0.7, 3.4
    Zmin, Zmax = -2.6, 2.6
    NR, NZ = 321, 321

    R = np.linspace(Rmin, Rmax, NR)
    Z = np.linspace(Zmin, Zmax, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # ---------------------------
    # 3) Compute vacuum psi and B
    # ---------------------------
    psi_vac, psi_basis = vacuum_psi_from_coils(R, Z, coils)
    BR, BZ = BR_BZ_from_psi(R, Z, psi_vac)
    Bpol = np.sqrt(BR**2 + BZ**2)

    # ---------------------------
    # 4) Create output directory
    # ---------------------------
    out_dir = next_design_dir(HARD_CODED_BASE, prefix="vacuum_design")
    plot_dir = os.path.join(out_dir, "plots")

    # ---------------------------
    # 5) Plots (sensible set)
    # ---------------------------

    # (a) Psi contours (flux surfaces) + coil locations
    plt.figure()
    cs = plt.contour(RR, ZZ, psi_vac, levels=40)
    plt.clabel(cs, inline=True, fontsize=7, fmt="%.1e")
    for c in coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
        plt.text(c["Rc"], c["Zc"], c["name"], fontsize=8)
    plt.axis("equal")
    plt.title("Vacuum poloidal flux surfaces (ψ contours)")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.grid(True, alpha=0.25)
    save_plot(os.path.join(plot_dir, "psi_contours.png"))

    # (b) Poloidal field magnitude map with flux contours overlaid
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bpol, shading="auto")
    plt.colorbar(im, label=r"$|B_p|$ [T] (from PF coils)")
    plt.contour(RR, ZZ, psi_vac, levels=25, linewidths=0.8)
    for c in coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
    plt.axis("equal")
    plt.title("Vacuum poloidal field magnitude |Bp| with flux contours")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bp_magnitude.png"))

    # (c) Streamplot of poloidal field (direction) on top of |Bp|
    # Streamplot expects X,Y as 2D in "xy" indexing. We'll transpose arrays accordingly.
    plt.figure()
    im = plt.pcolormesh(RR, ZZ, Bpol, shading="auto")
    plt.colorbar(im, label=r"$|B_p|$ [T]")
    # streamplot wants arrays shaped (NY,NX), so feed transposed:
    plt.streamplot(R, Z, BR.T, BZ.T, density=1.2, linewidth=0.7, arrowsize=1.0)
    for c in coils:
        plt.plot([c["Rc"]], [c["Zc"]], "o")
    plt.axis("equal")
    plt.title("Vacuum poloidal field lines (streamplot) over |Bp|")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    save_plot(os.path.join(plot_dir, "Bp_streamlines.png"))

    # ---------------------------
    # 6) Save HDF5
    # ---------------------------
    out_h5 = os.path.join(out_dir, HDF5_NAME)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("R", data=R)
        f.create_dataset("Z", data=Z)
        f.create_dataset("psi_vac", data=psi_vac)
        f.create_dataset("BR", data=BR)
        f.create_dataset("BZ", data=BZ)
        f.create_dataset("Bpol", data=Bpol)
        f.create_dataset("psi_basis_unit_current", data=psi_basis)  # helpful later for fitting

        g = f.create_group("coils")
        g.attrs["ncoils"] = len(coils)
        g.create_dataset("name", data=np.array([c["name"].encode("utf8") for c in coils]))
        g.create_dataset("shape", data=np.array([c.get("shape", "circular_filament").encode("utf8") for c in coils]))
        g.create_dataset("Rc", data=np.array([c["Rc"] for c in coils], float))
        g.create_dataset("Zc", data=np.array([c["Zc"] for c in coils], float))
        g.create_dataset("I",  data=np.array([c["I"]  for c in coils], float))

        f.attrs["notes"] = (
            "Vacuum PF-coil forward model. Coils are toroidal filament loops. "
            "psi_vac computed via elliptic integrals; BR,BZ from finite differences."
        )
        f.attrs["Rmin"] = float(Rmin); f.attrs["Rmax"] = float(Rmax)
        f.attrs["Zmin"] = float(Zmin); f.attrs["Zmax"] = float(Zmax)
        f.attrs["NR"] = int(NR); f.attrs["NZ"] = int(NZ)

    print(f"Vacuum forward-model outputs -> {out_dir}")
    print(f"  Data : {out_h5}")
    print(f"  Plots: {plot_dir}")


if __name__ == "__main__":
    main()

