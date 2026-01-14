#!/usr/bin/env python3
"""
Local stability diagnostics from postproc.h5 (your structure)
=============================================================

Reads:
  - postproc.h5 (hardcoded path)
  - gs_solution.h5 inferred from postproc location: ../../gs_solution.h5

Computes (standard & important local diagnostics):
  - r(psi) from midplane intersections: r = (R_out - R_in)/2
  - q(r), dq/dr
  - magnetic shear s(r) = (r/q) dq/dr
  - dp/dr (from fs_p vs r)
  - s-alpha ballooning proxy:
        alpha(r) = -(2*mu0*R0 * q^2 / <B>^2) * dp/dr
    where R0 is midplane center from intersections and <B> is fs_Bmag

Outputs:
  - <sim_dir>/local_stability/runXXX/stability.h5
  - plots/*.png
"""

import os
import re
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

MU0 = 4e-7 * np.pi


# --------------------------- output helpers --------------------------------

def next_run_dir(base_dir, prefix="run"):
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


def safe_gradient(y, x):
    """Gradient on finite values only; returns NaNs elsewhere."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    g = np.full_like(y, np.nan)

    ok = np.isfinite(y) & np.isfinite(x)
    if np.count_nonzero(ok) < 5:
        return g

    yy = y[ok]
    xx = x[ok]

    # Enforce increasing x for gradient
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]

    gy = np.gradient(yy, xx)

    # Map back
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    g_ok = gy[inv]
    g[ok] = g_ok
    return g


# ----------------------------- readers -------------------------------------

def load_postproc(post_path):
    with h5py.File(post_path, "r") as f:
        # Raw / convenience
        R = np.asarray(f["R"])
        Z = np.asarray(f["Z"])
        psi = np.asarray(f["psi"])
        inside_mask = np.asarray(f["inside_mask"]).astype(bool)

        # 1D profiles (your exact names)
        fs_psi = np.asarray(f["fs_psi"])
        psibar_levels = np.asarray(f["psibar_levels"])
        q = np.asarray(f["q_contour"])       # your q
        fs_Bmag = np.asarray(f["fs_Bmag"])
        fs_p = np.asarray(f["fs_p"])

        # Attributes (your exact names)
        R_axis = float(f.attrs["R_axis"])
        Z_axis = float(f.attrs["Z_axis"])
        psi_axis = float(f.attrs["psi_axis"])
        psi_b = float(f.attrs["psi_b"])
        R0_est = float(f.attrs.get("R0_est", np.nan))
        a_est = float(f.attrs.get("a_est", np.nan))

        # Optional diagnostics
        jphi = None
        if "gs_diagnostics" in f and "jphi" in f["gs_diagnostics"]:
            jphi = np.asarray(f["gs_diagnostics/jphi"])

        profile_params = {k: f.attrs[k] for k in f.attrs.keys() if k.startswith("profile_")}

    return dict(
        R=R, Z=Z, psi=psi, inside_mask=inside_mask,
        fs_psi=fs_psi, psibar_levels=psibar_levels, q=q, fs_Bmag=fs_Bmag, fs_p=fs_p,
        R_axis=R_axis, Z_axis=Z_axis, psi_axis=psi_axis, psi_b=psi_b, R0_est=R0_est, a_est=a_est,
        jphi=jphi, profile_params=profile_params
    )


def load_gs(gs_path):
    # For now we only validate it exists (you can expand later if needed)
    if not os.path.isfile(gs_path):
        raise FileNotFoundError(f"gs_solution.h5 not found at {gs_path}")
    return gs_path


# --------------------------- geometry core ---------------------------------

def midplane_intersections(R, Z, psi, inside_mask, psi_levels):
    """
    For each psi_level, find R_in and R_out at Z~0.
    Returns r = (R_out - R_in)/2 and R0 = (R_out + R_in)/2.
    """
    j_mid = int(np.argmin(np.abs(Z - 0.0)))
    psi_mid = psi[:, j_mid]
    mask_mid = inside_mask[:, j_mid]

    idx = np.where(mask_mid)[0]
    if idx.size < 5:
        raise ValueError("Inside-mask on midplane too small/nonexistent; cannot define r(psi).")

    iL, iR = idx[0], idx[-1]

    # Axis on midplane = minimum psi in the inside segment
    seg = psi_mid[iL:iR+1]
    ia = iL + int(np.argmin(seg))

    # Branches
    R_inb = R[iL:ia+1]
    psi_inb = psi_mid[iL:ia+1]
    R_outb = R[ia:iR+1]
    psi_outb = psi_mid[ia:iR+1]

    def invert_R_of_psi(Rb, psib, lvl):
        order = np.argsort(psib)
        ps = psib[order]
        Rs = Rb[order]
        if lvl < ps.min() or lvl > ps.max():
            return np.nan
        return np.interp(lvl, ps, Rs)

    R_in = np.array([invert_R_of_psi(R_inb, psi_inb, lvl) for lvl in psi_levels], dtype=float)
    R_out = np.array([invert_R_of_psi(R_outb, psi_outb, lvl) for lvl in psi_levels], dtype=float)

    good = np.isfinite(R_in) & np.isfinite(R_out) & (R_out > R_in)

    R0 = 0.5 * (R_in + R_out)
    r  = 0.5 * (R_out - R_in)

    return R_in, R_out, R0, r, good, j_mid


# ------------------------------ main ---------------------------------------

def main():
    # ---------------------------------------------------------------------
    # Hardcoded postproc path (your request)
    # ---------------------------------------------------------------------
    post_path = os.path.abspath(
        "/Users/leon/Desktop/python_skripte/tokamak_design/runs/sim001/postproc/run001/postproc.h5"
    )
    if not os.path.isfile(post_path):
        raise FileNotFoundError(f"postproc.h5 not found at {post_path}")

    # gs_solution.h5 is two levels above: runXXX -> postproc -> simXXX
    gs_path = os.path.abspath(os.path.join(os.path.dirname(post_path), "..", "..", "gs_solution.h5"))
    load_gs(gs_path)

    data = load_postproc(post_path)

    R = data["R"]; Z = data["Z"]; psi = data["psi"]; inside_mask = data["inside_mask"]
    fs_psi = data["fs_psi"]
    q = data["q"]
    fs_p = data["fs_p"]
    fs_Bmag = data["fs_Bmag"]
    psibar_levels = data["psibar_levels"]

    # Output directory: <sim_dir>/local_stability/runXXX
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(post_path), "..", ".."))
    out_dir = next_run_dir(os.path.join(sim_dir, "local_stability"), prefix="run")
    plot_dir = os.path.join(out_dir, "plots")

    print("Local stability analysis")
    print(f"  GS solution : {gs_path}")
    print(f"  Postproc    : {post_path}")
    print(f"  Output      : {out_dir}")

    # Geometry vs psi from midplane intersections
    R_in, R_out, R0, r, good, j_mid = midplane_intersections(R, Z, psi, inside_mask, fs_psi)

    # Derivatives vs r
    dq_dr = safe_gradient(q, r)
    s = (r / (q + 1e-30)) * dq_dr

    dp_dr = safe_gradient(fs_p, r)

    # s-alpha proxy (standard diagnostic)
    alpha = -(2.0 * MU0 * R0 * (q**2) / (fs_Bmag**2 + 1e-30)) * dp_dr

    # Mask invalid points
    def m(x):
        y = np.asarray(x, float).copy()
        y[~good] = np.nan
        return y

    r_m = m(r); q_m = m(q); s_m = m(s); dq_dr_m = m(dq_dr)
    p_m = m(fs_p); dp_dr_m = m(dp_dr); B_m = m(fs_Bmag); alpha_m = m(alpha)
    R0_m = m(R0); Rin_m = m(R_in); Rout_m = m(R_out)
    psibar_m = m(psibar_levels)

    # ---------------------------- PLOTS ---------------------------------

    # Midplane psi(R)
    plt.figure()
    plt.plot(R, psi[:, j_mid])
    plt.title("Midplane psi(R)")
    plt.xlabel("R [m]")
    plt.ylabel("psi")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "midplane_psi_R.png"))

    # r(psi) sanity plot
    plt.figure()
    plt.plot(psibar_m, r_m, marker="o", markersize=3, linewidth=1)
    plt.title("Minor radius r vs normalized flux")
    plt.xlabel(r"$\bar\psi$")
    plt.ylabel("r [m]")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "r_vs_psibar.png"))

    # q(r)
    plt.figure()
    plt.plot(r_m, q_m, marker="o", markersize=3, linewidth=1)
    plt.title("q(r)")
    plt.xlabel("r [m]")
    plt.ylabel("q")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "q_vs_r.png"))

    # shear s(r)
    plt.figure()
    plt.plot(r_m, s_m, marker="o", markersize=3, linewidth=1)
    plt.title("Magnetic shear s(r) = (r/q) dq/dr")
    plt.xlabel("r [m]")
    plt.ylabel("s")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "shear_s_vs_r.png"))

    # p(r), dp/dr
    plt.figure()
    plt.plot(r_m, p_m, marker="o", markersize=3, linewidth=1)
    plt.title("Flux-surface pressure p(r)")
    plt.xlabel("r [m]")
    plt.ylabel("p")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "p_vs_r.png"))

    plt.figure()
    plt.plot(r_m, dp_dr_m, marker="o", markersize=3, linewidth=1)
    plt.title("dp/dr vs r")
    plt.xlabel("r [m]")
    plt.ylabel("dp/dr")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "dpdr_vs_r.png"))

    # alpha(r)
    plt.figure()
    plt.plot(r_m, alpha_m, marker="o", markersize=3, linewidth=1)
    plt.title(r"Ballooning proxy $\alpha(r)$")
    plt.xlabel("r [m]")
    plt.ylabel(r"$\alpha$")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "alpha_vs_r.png"))

    # s-alpha diagram
    plt.figure()
    plt.scatter(s_m, alpha_m, s=12)
    plt.title("Local s–alpha diagram (proxy)")
    plt.xlabel("s")
    plt.ylabel("alpha")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(plot_dir, "s_alpha.png"))

    # ---------------------------- SAVE H5 --------------------------------

    out_h5 = os.path.join(out_dir, "stability.h5")
    with h5py.File(out_h5, "w") as f:
        f.attrs["GS_FILE"] = gs_path
        f.attrs["POSTPROC_FILE"] = post_path
        f.attrs["notes"] = (
            "Local stability diagnostics: r from midplane intersections, "
            "shear s=(r/q)dq/dr, and s–alpha ballooning proxy."
        )

        # Copy some equilibrium scalars from postproc attrs for convenience
        f.attrs["R_axis"] = data["R_axis"]
        f.attrs["Z_axis"] = data["Z_axis"]
        f.attrs["psi_axis"] = data["psi_axis"]
        f.attrs["psi_b"] = data["psi_b"]
        f.attrs["R0_est"] = data["R0_est"]
        f.attrs["a_est"] = data["a_est"]

        # Save profile params if present
        for k, v in data["profile_params"].items():
            f.attrs[k] = float(v)

        g = f.create_group("profiles")
        g.create_dataset("fs_psi", data=fs_psi)
        g.create_dataset("psibar_levels", data=psibar_levels)

        g.create_dataset("R_in_midplane", data=R_in)
        g.create_dataset("R_out_midplane", data=R_out)
        g.create_dataset("R0_midplane", data=R0)
        g.create_dataset("r_minor", data=r)
        g.create_dataset("good", data=good.astype(np.uint8))

        g.create_dataset("q", data=q)
        g.create_dataset("dq_dr", data=dq_dr)
        g.create_dataset("shear_s", data=s)

        g.create_dataset("p", data=fs_p)
        g.create_dataset("dp_dr", data=dp_dr)
        g.create_dataset("Bmag_fs", data=fs_Bmag)
        g.create_dataset("alpha", data=alpha)

    print(f"Saved: {out_h5}")
    print(f"Plots in: {plot_dir}")


if __name__ == "__main__":
    main()
