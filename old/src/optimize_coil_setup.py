#!/usr/bin/env python3
import os
import csv
import re
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# IMPORTANT: import your solver module
# Ensure GS_free_boundary_solver.py is importable (same dir or in PYTHONPATH)
import GS_free_boundary_solver as gs


# -------------------------- hardcoded IO --------------------------
HARD_CODED_BASE = "/Users/leon/Desktop/python_skripte/tokamak_design/old/runs"

COILS_CSV  = "/Users/leon/Desktop/python_skripte/tokamak_design/old/inputs/pf_coils.csv"
PARAMS_CSV  = "/Users/leon/Desktop/python_skripte/tokamak_design/old/inputs/gs_force_balance_params.csv"
TARGETS_CSV = "/Users/leon/Desktop/python_skripte/tokamak_design/old/inputs/targets.csv"

OUT_PREFIX = "optimize_equilibrium"


# -------------------------- defaults for targets --------------------------
TARGET_DEFAULTS = dict(
    # target plasma boundary (D-shape)
    R0=1.65,
    a=0.50,
    kappa=1.7,
    delta=0.35,

    # X-point target (optional)
    want_xpoint=1,         # 1 = include, 0 = ignore
    Rx=1.90,
    Zx=-1.50,

    # axis target (optional but helpful)
    want_axis=1,
    Raxis=1.65,
    Zaxis=0.0,

    # boundary constraints
    n_boundary=40,         # points around boundary
    w_boundary=1.0,        # weight
    w_xpoint=3.0,          # weight
    w_axis=2.0,            # weight

    # optimization
    max_outer=15,
    dI_fd=2000.0,          # finite-difference perturbation in A
    step_scale=0.5,        # damping on update
    reg_lambda=1e-4,       # Tikhonov regularization
    tol_rms=1e-3,          # stop when RMS residual small

    # current limits (optional)
    Imax=5e5,
)

DEFAULT_COILS = [
    ("OB_top", 2.30, +1.10, +2.0e5),
    ("OB_bot", 2.30, -1.10, +2.0e5),
    ("IB_top", 1.20, +0.95, -1.4e5),
    ("IB_bot", 1.20, -0.95, -1.4e5),
    ("VF_top", 2.80, +2.10, +0.6e5),
    ("VF_bot", 2.80, -2.10, +0.6e5),
]

# -------------------------- small helpers --------------------------

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


def read_keyval_csv(path, defaults):
    d = dict(defaults)
    if not os.path.isfile(path):
        print(f"[info] targets CSV missing -> using defaults: {path}")
        return d
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 2:
                continue
            k = row[0].strip()
            v = row[1].strip()
            if not k or k.startswith("#"):
                continue
            try:
                d[k] = float(v)
            except Exception:
                d[k] = v
    return d


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


def read_params_csv(path):
    # reuse your solver's CSV reading if you want; here minimal:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing params CSV: {path}")
    params = {}
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) < 2: 
                continue
            k = row[0].strip()
            v = row[1].strip()
            if not k or k.startswith("#"):
                continue
            if k in ("NR", "NZ", "max_iter"):
                params[k] = int(float(v))
            else:
                params[k] = float(v)
    return params


def dshape_boundary(R0, a, kappa, delta, n=40):
    """
    Standard tokamak D-shape parameterization.
      R(θ) = R0 + a cos(θ + δ sin θ)
      Z(θ) = κ a sin θ
    """
    th = np.linspace(0, 2*np.pi, n, endpoint=False)
    Rb = R0 + a*np.cos(th + delta*np.sin(th))
    Zb = kappa*a*np.sin(th)
    return Rb, Zb


def residual_vector(res, tgt):
    """
    Build residual vector r and weights.
    r contains:
      - boundary psi mismatch: psi(Rb,Zb) - psi_sep
      - (optional) x-point position mismatch
      - (optional) axis position mismatch
    """
    R = res["R"]; Z = res["Z"]; psi = res["psi"]
    psi_sep = res["psi_sep"]

    Rb, Zb = dshape_boundary(tgt["R0"], tgt["a"], tgt["kappa"], tgt["delta"], n=int(tgt["n_boundary"]))
    interp_psi = RegularGridInterpolator((R, Z), psi, bounds_error=False, fill_value=np.nan)

    pts = np.vstack([Rb, Zb]).T
    psi_b = interp_psi(pts)

    # boundary residual: want psi(Rb,Zb) == psi_sep
    rb = (psi_b - psi_sep)
    rb = np.nan_to_num(rb, nan=0.0)

    r_list = [np.sqrt(tgt["w_boundary"]) * rb]

    # x-point residual (if requested and found)
    if int(tgt["want_xpoint"]) == 1:
        if res["used_xpoint"]:
            rx = np.array([res["xpoint_R"] - tgt["Rx"], res["xpoint_Z"] - tgt["Zx"]], float)
        else:
            # penalize "no x-point" by treating as large mismatch
            rx = np.array([5.0, 5.0], float)  # meters (big penalty)
        r_list.append(np.sqrt(tgt["w_xpoint"]) * rx)

    # axis residual
    if int(tgt["want_axis"]) == 1:
        ra = np.array([res["axis_R"] - tgt["Raxis"], res["axis_Z"] - tgt["Zaxis"]], float)
        r_list.append(np.sqrt(tgt["w_axis"]) * ra)

    r = np.concatenate(r_list)
    return r


def solve_update(J, r, lam):
    """
    Solve (J^T J + lam I) dI = -J^T r
    """
    JTJ = J.T @ J
    A = JTJ + lam * np.eye(JTJ.shape[0])
    b = -(J.T @ r)
    return np.linalg.solve(A, b)


# -------------------------- main optimization --------------------------

def main():
    params = read_params_csv(PARAMS_CSV)
    coils = read_coils_csv(COILS_CSV)
    tgt = read_keyval_csv(TARGETS_CSV, TARGET_DEFAULTS)

    out_dir = next_run_dir(HARD_CODED_BASE, OUT_PREFIX)
    plot_dir = os.path.join(out_dir, "plots")

    ncoils = len(coils)
    I = np.array([c["I"] for c in coils], float)

    hist = []

    for it in range(1, int(tgt["max_outer"]) + 1):
        # run forward GS
        for k in range(ncoils):
            coils[k]["I"] = float(I[k])

        res = gs.run_gs_force_balance(params, coils,
                                      make_plots=False, out_dir=None)

        r0 = residual_vector(res, tgt)
        rms = float(np.sqrt(np.mean(r0**2)))
        print(f"[outer {it:02d}] rms residual = {rms:.3e}   used_xpoint={res['used_xpoint']}")

        hist.append(dict(iter=it, rms=rms, used_xpoint=int(res["used_xpoint"]), I=I.copy()))

        if rms < float(tgt["tol_rms"]):
            print("Converged.")
            break

        # build Jacobian by finite differences wrt coil currents
        dI = float(tgt["dI_fd"])
        J = np.zeros((len(r0), ncoils), float)

        for k in range(ncoils):
            I_pert = I.copy()
            I_pert[k] += dI
            for kk in range(ncoils):
                coils[kk]["I"] = float(I_pert[kk])

            resk = gs.run_gs_force_balance(params, coils,
                                           make_plots=False, out_dir=None)
            rk = residual_vector(resk, tgt)
            J[:, k] = (rk - r0) / dI

        # compute update
        lam = float(tgt["reg_lambda"])
        dI_update = solve_update(J, r0, lam)

        # damp step
        I = I + float(tgt["step_scale"]) * dI_update

        # clamp
        Imax = float(tgt["Imax"])
        I = np.clip(I, -Imax, Imax)

    # Save final coils CSV
    coils_out = os.path.join(out_dir, "pf_coils_optimized.csv")
    with open(coils_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "Rc", "Zc", "I"])
        for k, c in enumerate(coils):
            w.writerow([c["name"], c["Rc"], c["Zc"], float(I[k])])

    # Save history to HDF5
    out_h5 = os.path.join(out_dir, "optimization.h5")
    with h5py.File(out_h5, "w") as f:
        f.attrs["PARAMS_CSV"] = PARAMS_CSV
        f.attrs["COILS_CSV_INIT"] = COILS_CSV
        f.attrs["TARGETS_CSV"] = TARGETS_CSV
        f.attrs["ncoils"] = ncoils

        iters = np.array([h["iter"] for h in hist], int)
        rmsv  = np.array([h["rms"] for h in hist], float)
        used  = np.array([h["used_xpoint"] for h in hist], int)
        I_hist = np.stack([h["I"] for h in hist], axis=0)

        f.create_dataset("iter", data=iters)
        f.create_dataset("rms", data=rmsv)
        f.create_dataset("used_xpoint", data=used)
        f.create_dataset("I_hist", data=I_hist)

    # Make one final forward solve with plots and save to its own subfolder
    final_dir = os.path.join(out_dir, "final_equilibrium")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(os.path.join(final_dir, "plots"), exist_ok=True)

    for k in range(ncoils):
        coils[k]["I"] = float(I[k])

    res_final = gs.run_gs_force_balance(params, coils,
                                       make_plots=True, out_dir=final_dir)

    # Plot optimization history
    plt.figure()
    plt.plot(np.arange(len(rmsv)), rmsv, marker="o")
    plt.yscale("log")
    plt.xlabel("outer iteration")
    plt.ylabel("RMS residual")
    plt.grid(True, alpha=0.3)
    plt.title("Optimization convergence")
    plt.savefig(os.path.join(plot_dir, "rms_history.png"), dpi=180)
    plt.close()

    print(f"Saved optimization run -> {out_dir}")
    print(f"  coils: {coils_out}")
    print(f"  hist : {out_h5}")
    print(f"  final equilibrium folder: {final_dir}")


if __name__ == "__main__":
    main()




