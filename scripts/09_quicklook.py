#!/usr/bin/env python3
"""
09_quicklook.py
===============

Quicklook orchestrator for the *new workflow state* (Stage 01 implemented).

What it does
------------
• Reads from results.h5 using tokdesign.io.h5 (single I/O layer)
• Uses tokdesign.viz plotting modules (pure plotting, no HDF5 inside viz)
• Saves figures under run_dir/figures/ in subdirectories by content bucket.

Expected HDF5 layout (Stage 01)
-------------------------------
Stage 01 outputs live under:
  /stage01_fixed/...

In particular:
  /stage01_fixed/grid/{R,Z,RR,ZZ}
  /stage01_fixed/best/equilibrium/{psi, psi_axis, psi_lcfs, lcfs/*, plasma_mask, j_phi, fields/*}
  /stage01_fixed/best/profiles/{psi_bar, rho, p, F, q, s, alpha}
  /stage01_fixed/best/metrics/*            (scalars)
  /stage01_fixed/trace/{objective_total, feasible, constraints/margins, ...}

Output structure
----------------
run_dir/figures/
  10_equilibrium/
  11_fields/
  12_profiles/
  13_derived/

Usage
-----
python scripts/09_quicklook.py --run-dir data/runs/<RUN_ID> --formats png pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from tokdesign.io.h5 import open_h5, h5_read_array, h5_read_scalar
from tokdesign.io.logging_utils import setup_logger
from tokdesign.viz.style import apply_mpl_defaults, run_header

from tokdesign.viz.plot_equilibrium import (
    plot_psi_map,
    plot_scalar_map,
    plot_midplane_cuts,
)
from tokdesign.viz.plot_profiles import plot_profiles_vs_psin
from tokdesign.viz.plot_fieldlines import plot_fieldlines_3d
from tokdesign.viz.plot_metrics import plot_radial_profiles


STAGE = "/stage01_fixed"


def _maybe(h5, path: str) -> bool:
    return path in h5


def _decode_strings(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in np.ravel(arr):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


def save_figure(fig: plt.Figure, out_base: Path, formats: Sequence[str], dpi: int = 160) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def _read_opt_array(h5, path: str) -> Optional[np.ndarray]:
    if not _maybe(h5, path):
        return None
    try:
        return np.asarray(h5_read_array(h5, path))
    except Exception:
        return None


def _read_opt_scalar(h5, path: str) -> Any:
    if not _maybe(h5, path):
        return None
    try:
        return h5_read_scalar(h5, path)
    except Exception:
        return None


def _read_vessel_boundary(h5) -> Optional[np.ndarray]:
    # Device exists from earlier stages in the run folder, depending on what you already ran.
    for path in ("/device/vessel_boundary", "/device/vessel/boundary"):
        vb = _read_opt_array(h5, path)
        if vb is not None and vb.ndim == 2 and vb.shape[1] == 2 and vb.shape[0] >= 3:
            return vb
    return None


def _make_derived_summary_figure(run_id: str, schema_version: str, metrics: Dict[str, Any]) -> plt.Figure:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(f"Run: {run_id}   (schema {schema_version})\nDerived summary (best/metrics)")

    if not metrics:
        ax.text(0.03, 0.92, "(no /stage01_fixed/best/metrics/* found)", va="top", fontsize=12)
        return fig

    # format lines
    keys = sorted(metrics.keys())
    lines = []
    for k in keys:
        v = metrics[k]
        try:
            vv = float(v)
            if not np.isfinite(vv):
                s = "NaN"
            else:
                av = abs(vv)
                s = f"{vv:.3e}" if (av != 0 and (av < 1e-3 or av >= 1e4)) else f"{vv:.6g}"
        except Exception:
            s = str(v)
        lines.append(f"{k} = {s}")

    ax.text(0.03, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")
    return fig


def _pick_fieldline_seeds(R: np.ndarray, Z: np.ndarray, mask: Optional[np.ndarray]) -> List[tuple]:
    # Very simple: midplane points inside mask if available; else center.
    if mask is None:
        return [(float(R[len(R)//2]), 0.0)]

    mask = np.asarray(mask, bool)
    iz0 = int(np.argmin(np.abs(Z - 0.0)))

    inside = mask[iz0, :]
    idx = np.where(inside)[0]
    if idx.size < 5:
        return [(float(R[len(R)//2]), float(Z[iz0]))]

    # 4 seeds across the plasma
    lo = int(idx.min() + 0.10 * (idx.max() - idx.min()))
    hi = int(idx.min() + 0.90 * (idx.max() - idx.min()))
    irs = np.linspace(lo, hi, 4).astype(int).tolist()
    return [(float(R[ir]), float(Z[iz0])) for ir in irs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard quicklook plots for a run (Stage 01 only).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--run-dir", required=True, help="Run directory containing results.h5 and figures/")
    parser.add_argument("--formats", nargs="+", default=["png"], help="e.g. png pdf")
    parser.add_argument("--dpi", type=int, default=160, help="DPI for raster outputs")

    # Fieldline controls (optional)
    parser.add_argument("--fieldlines", action="store_true", help="Also plot exemplary 3D field lines if fields exist")
    parser.add_argument("--fieldline-steps", type=int, default=2500, help="Fieldline integration steps")
    parser.add_argument("--fieldline-ds", type=float, default=0.01, help="Fieldline ds (pseudo arclength step)")
    args = parser.parse_args()

    apply_mpl_defaults()

    run_dir = Path(args.run_dir).expanduser().resolve()
    h5_path = run_dir / "results.h5"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories (matching the report content buckets)
    d10 = fig_dir / "10_equilibrium"
    d11 = fig_dir / "11_fields"
    d12 = fig_dir / "12_profiles"
    d13 = fig_dir / "13_derived"

    if not h5_path.exists():
        raise FileNotFoundError(f"results.h5 not found: {h5_path}")

    log_path = run_dir / "run.log"
    logger = setup_logger(log_path, level=args.log_level)

    logger.info("========== Stage 09: Quicklook (figures) ==========")
    logger.info("run_dir: %s", str(run_dir))

    # -----------------------------
    # Load datasets via io/h5.py
    # -----------------------------
    with open_h5(h5_path, "r") as h5:
        schema_version = str(_read_opt_scalar(h5, "/meta/schema_version") or "unknown")
        run_id = str(_read_opt_scalar(h5, "/meta/run_id") or run_dir.name)

        if STAGE not in h5:
            raise KeyError(f"Missing {STAGE} in {h5_path}. Did Stage 01 write outputs?")

        # Optional device overlay
        vessel = _read_vessel_boundary(h5)

        # Grid
        R = _read_opt_array(h5, f"{STAGE}/grid/R")
        Z = _read_opt_array(h5, f"{STAGE}/grid/Z")

        # Best equilibrium
        psi = _read_opt_array(h5, f"{STAGE}/best/equilibrium/psi")
        psi_axis = _read_opt_scalar(h5, f"{STAGE}/best/equilibrium/psi_axis")
        psi_lcfs = _read_opt_scalar(h5, f"{STAGE}/best/equilibrium/psi_lcfs")

        lcfs_R = _read_opt_array(h5, f"{STAGE}/best/equilibrium/lcfs/R")
        lcfs_Z = _read_opt_array(h5, f"{STAGE}/best/equilibrium/lcfs/Z")
        lcfs_boundary = None
        if lcfs_R is not None and lcfs_Z is not None and lcfs_R.size >= 3 and lcfs_Z.size >= 3:
            lcfs_boundary = np.column_stack([lcfs_R.ravel(), lcfs_Z.ravel()])

        plasma_mask = _read_opt_array(h5, f"{STAGE}/best/equilibrium/plasma_mask")
        if plasma_mask is not None:
            plasma_mask = np.asarray(plasma_mask).astype(bool)

        jphi = _read_opt_array(h5, f"{STAGE}/best/equilibrium/j_phi")

        BR = _read_opt_array(h5, f"{STAGE}/best/equilibrium/fields/BR")
        BZ = _read_opt_array(h5, f"{STAGE}/best/equilibrium/fields/BZ")
        Bphi = _read_opt_array(h5, f"{STAGE}/best/equilibrium/fields/Bphi")

        # 1D profiles
        rho = _read_opt_array(h5, f"{STAGE}/best/profiles/rho")
        p1 = _read_opt_array(h5, f"{STAGE}/best/profiles/p")
        F1 = _read_opt_array(h5, f"{STAGE}/best/profiles/F")
        q1 = _read_opt_array(h5, f"{STAGE}/best/profiles/q")
        s1 = _read_opt_array(h5, f"{STAGE}/best/profiles/s")
        a1 = _read_opt_array(h5, f"{STAGE}/best/profiles/alpha")

        # Traces
        obj = _read_opt_array(h5, f"{STAGE}/trace/objective_total")
        feas = _read_opt_array(h5, f"{STAGE}/trace/feasible")
        margins = _read_opt_array(h5, f"{STAGE}/trace/constraints/margins")

        # Best metrics (scalars)
        metrics: Dict[str, Any] = {}
        metrics_grp = f"{STAGE}/best/metrics"
        if _maybe(h5, metrics_grp):
            try:
                for k in h5[metrics_grp].keys():
                    metrics[k] = _read_opt_scalar(h5, f"{metrics_grp}/{k}")
            except Exception:
                metrics = {}

    # Basic sanity
    if R is None or Z is None:
        raise KeyError(f"Missing grid arrays at {STAGE}/grid/R or {STAGE}/grid/Z")
    if psi is None:
        raise KeyError(f"Missing best equilibrium psi at {STAGE}/best/equilibrium/psi")

    # -----------------------------
    # 10: equilibrium
    # -----------------------------
    fig, _ = plot_psi_map(
        run_id=run_id,
        schema_version=schema_version,
        R=R,
        Z=Z,
        psi=psi,
        psi_axis=float(psi_axis) if psi_axis is not None else None,
        psi_lcfs=float(psi_lcfs) if psi_lcfs is not None else None,
        vessel_boundary=vessel,
        lcfs_boundary=lcfs_boundary,
    )
    save_figure(fig, d10 / "psi_map", args.formats, dpi=args.dpi)
    plt.close(fig)
    logger.info("plotted psi_map and saved as: %s ", d10 / "psi_map")

    if jphi is not None:
        fig, _ = plot_scalar_map(
            run_id=run_id,
            schema_version=schema_version,
            title="Toroidal current density jφ(R,Z)",
            R=R,
            Z=Z,
            field=jphi,
            cbar_label="jφ [A/m²]",
            vessel_boundary=vessel,
            lcfs_boundary=lcfs_boundary,
            mask=plasma_mask,
        )
        save_figure(fig, d10 / "jphi_map", args.formats, dpi=args.dpi)
        plt.close(fig)
        logger.info("plotted jphi_map and saved as: %s ", d10 / "jphi_map")

    # -----------------------------
    # 11: fields + midplane cuts
    # -----------------------------
    if BR is not None and BZ is not None:
        Bp = np.sqrt(np.asarray(BR, float) ** 2 + np.asarray(BZ, float) ** 2)
        fig, _ = plot_scalar_map(
            run_id=run_id,
            schema_version=schema_version,
            title="Poloidal field magnitude Bp(R,Z)",
            R=R,
            Z=Z,
            field=Bp,
            cbar_label="Bp [T]",
            vessel_boundary=vessel,
            lcfs_boundary=lcfs_boundary,
            mask=None,
        )
        save_figure(fig, d11 / "Bp_map", args.formats, dpi=args.dpi)
        plt.close(fig)
        logger.info("plotted Bp_map and saved as: %s ", d11 / "Bp_map")

    if BR is not None and BZ is not None and Bphi is not None:
        Bmag = np.sqrt(np.asarray(BR, float) ** 2 + np.asarray(BZ, float) ** 2 + np.asarray(Bphi, float) ** 2)
        fig, _ = plot_scalar_map(
            run_id=run_id,
            schema_version=schema_version,
            title="Total field magnitude |B|(R,Z)",
            R=R,
            Z=Z,
            field=Bmag,
            cbar_label="|B| [T]",
            vessel_boundary=vessel,
            lcfs_boundary=lcfs_boundary,
            mask=None,
        )
        save_figure(fig, d11 / "Bmag_map", args.formats, dpi=args.dpi)
        plt.close(fig)
        logger.info("plotted Bmag_map and saved as: %s ", d11 / "Bmag_map")

    fig, _ = plot_midplane_cuts(
        run_id=run_id,
        schema_version=schema_version,
        R=R,
        Z=Z,
        psi=psi,
        BR=BR,
        BZ=BZ,
        Bphi=Bphi,
        jphi=jphi,
        p=None,
        z0=0.0,
    )
    save_figure(fig, d11 / "midplane_cuts", args.formats, dpi=args.dpi)
    plt.close(fig)
    logger.info("plotted midplane_cuts and saved as: %s ", d11 / "midplane_cuts")

    # Optional 3D field lines
    if args.fieldlines and (BR is not None) and (BZ is not None) and (Bphi is not None):
        seeds = _pick_fieldline_seeds(R, Z, plasma_mask)
        fig = plot_fieldlines_3d(
            run_id=run_id,
            schema_version=schema_version,
            R=R,
            Z=Z,
            BR=BR,
            BZ=BZ,
            Bphi=Bphi,
            seeds=seeds,
            ds=float(args.fieldline_ds),
            nsteps=int(args.fieldline_steps),
        )
        save_figure(fig, d11 / "fieldlines_3d", args.formats, dpi=args.dpi)
        plt.close(fig)
        logger.info("plotted fieldlines_3d and saved as: %s ", d11 / "fieldlines_3d")

    # -----------------------------
    # 12: profiles
    # -----------------------------
    # Binned profiles vs ψ̄ using grid psi + mask (very useful debug plot)
    if (psi_axis is not None) and (psi_lcfs is not None) and (plasma_mask is not None):
        denom = float(psi_lcfs) - float(psi_axis)
        if abs(denom) > 0:
            psin = (np.asarray(psi, float) - float(psi_axis)) / denom
            psin = np.clip(psin, 0.0, 1.0)

            Bp2 = None
            Bmag2 = None
            if BR is not None and BZ is not None:
                Bp2 = np.sqrt(np.asarray(BR, float) ** 2 + np.asarray(BZ, float) ** 2)
            if BR is not None and BZ is not None and Bphi is not None:
                Bmag2 = np.sqrt(np.asarray(BR, float) ** 2 + np.asarray(BZ, float) ** 2 + np.asarray(Bphi, float) ** 2)

            fig, _ = plot_profiles_vs_psin(
                run_id=run_id,
                schema_version=schema_version,
                psin=psin,
                mask=plasma_mask,
                p=None,        # stage01_fixed stores p as 1D profile; no 2D p field
                F=None,        # stage01_fixed stores F as 1D profile; no 2D F field
                jphi=jphi,
                Bp=Bp2,
                Bmag=Bmag2,
                nbins=45,
            )
            save_figure(fig, d12 / "profiles_vs_psin_binned", args.formats, dpi=args.dpi)
            plt.close(fig)
            logger.info("plotted profiles_vs_psin_binned and saved as: %s ", d12 / "profiles_vs_psin_binned")

    # 1D profiles vs rho (from best/profiles/*)
    if rho is not None:
        fig, _axes = plot_radial_profiles(
            run_id=run_id,
            schema_version=schema_version,
            rho=rho,
            p=p1,
            F=F1,
            q=q1,
            s=s1,
            alpha=a1,
        )
        save_figure(fig, d12 / "profiles_1d_vs_rho", args.formats, dpi=args.dpi)
        plt.close(fig)
        logger.info("plotted profiles_1d_vs_rho and saved as: %s ", d12 / "profiles_1d_vs_rho")

    # -----------------------------
    # 13: derived + optimization traces
    # -----------------------------
    fig = _make_derived_summary_figure(run_id, schema_version, metrics)
    save_figure(fig, d13 / "derived_summary_metrics", args.formats, dpi=args.dpi)
    plt.close(fig)
    logger.info("plotted derived_summary_metrics and saved as: %s ", d13 / "derived_summary_metrics")

    # Optimization traces (objective, feasible, min constraint margin)
    if obj is not None or feas is not None or margins is not None:
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        run_header(ax, run_id=run_id, schema_version=schema_version, subtitle="Optimization traces (Stage 01)")
        ax.set_xlabel("eval")
        ax.grid(True, alpha=0.25)

        n = 0
        for arr in (obj, feas, margins):
            if arr is None:
                continue
            try:
                n = max(n, int(np.asarray(arr).shape[0]))
            except Exception:
                pass
        x = np.arange(n)

        if obj is not None:
            y = np.asarray(obj, float).ravel()
            ax.plot(x[: y.shape[0]], y, lw=2.0, label="objective_total")
        if feas is not None:
            y = np.asarray(feas, float).ravel()
            ax.plot(x[: y.shape[0]], y, lw=1.6, label="feasible (0/1)")
        if margins is not None:
            mm = np.asarray(margins, float)
            if mm.ndim == 2 and mm.shape[0] > 0:
                y = np.nanmin(mm, axis=1)
                ax.plot(x[: y.shape[0]], y, lw=1.6, label="min constraint margin")

        ax.legend(loc="best")
        save_figure(fig, d13 / "traces", args.formats, dpi=args.dpi)
        plt.close(fig)
        logger.info("plotted traces and saved as: %s ", d13 / "traces")

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\nQuicklook plots saved to:")
    print(f"  {fig_dir}")
    print("Subdirectories written:")
    print("  10_equilibrium, 11_fields, 12_profiles, 13_derived\n")


if __name__ == "__main__":
    main()
