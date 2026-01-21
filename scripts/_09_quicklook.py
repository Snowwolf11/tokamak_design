#!/usr/bin/env python3
"""
09_quicklook.py
===============

Orchestrator script:
• Reads from results.h5 using tokdesign.io.h5 (single I/O layer)
• Calls tokdesign.viz plotting functions
• Saves figures to run_dir/figures/ in subdirectories by stage/content.

Output structure
----------------
run_dir/figures/
  01_device/
  02_target/
  03_greens/
  04_coil_fit/
  10_equilibrium/
  11_fields/
  12_profiles/
  13_derived/

Usage
-----
python scripts/09_quicklook.py --run-dir data/runs/<RUN_ID> --formats png pdf --greens-max 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import matplotlib.pyplot as plt

from tokdesign.io.h5 import open_h5, h5_read_array, h5_read_scalar
from tokdesign.viz.style import apply_mpl_defaults
from tokdesign.viz.plot_device import plot_device_geometry
from tokdesign.viz.plot_target import plot_target_overlay
from tokdesign.viz.plot_greens import plot_coil_greens

from tokdesign.viz.plot_equilibrium import plot_psi_map, plot_scalar_map, plot_midplane_cuts
from tokdesign.viz.plot_profiles import plot_profiles_vs_psin
from tokdesign.viz.plot_fieldlines import plot_fieldlines_3d

# NEW (04)
from tokdesign.viz.plot_coil_fit import (
    plot_boundary_psi_vs_angle,
    plot_boundary_dpsi,
    plot_currents_vs_limits,
    compute_psi_vacuum,
    plot_vacuum_flux_surfaces_compare,
    plot_vacuum_psi_map,
)


def _decode_strings(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in arr:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def save_figure(fig: plt.Figure, out_base: Path, formats: Sequence[str], dpi: int = 160) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        fig.savefig(out_base.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def _maybe(h5, path: str) -> bool:
    return path in h5


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standard plots for a run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing results.h5 and figures/")
    parser.add_argument("--formats", nargs="+", default=["png"], help="e.g. png pdf")
    parser.add_argument("--dpi", type=int, default=160, help="DPI for raster outputs")
    parser.add_argument("--coils", nargs="*", default=None, help="Coil names to plot greens for (default: first few)")
    parser.add_argument("--greens-max", type=int, default=4, help="Max number of greens plots")
    parser.add_argument("--greens-levels", type=int, default=30, help="Number of contour levels for greens")

    # Fieldline controls (simple)
    parser.add_argument("--fieldlines", action="store_true", help="Also plot exemplary 3D field lines if fields exist")
    parser.add_argument("--fieldline-steps", type=int, default=2500, help="Fieldline integration steps")
    parser.add_argument("--fieldline-ds", type=float, default=0.01, help="Fieldline ds (pseudo arclength step)")
    args = parser.parse_args()

    apply_mpl_defaults()

    run_dir = Path(args.run_dir).expanduser().resolve()
    h5_path = run_dir / "results.h5"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    d01 = fig_dir / "01_device"
    d02 = fig_dir / "02_target"
    d03 = fig_dir / "03_greens"
    d04 = fig_dir / "04_coil_fit"
    d10 = fig_dir / "10_equilibrium"
    d11 = fig_dir / "11_fields"
    d12 = fig_dir / "12_profiles"
    d13 = fig_dir / "13_derived"

    if not h5_path.exists():
        raise FileNotFoundError(f"results.h5 not found: {h5_path}")

    # -----------------------------
    # Load datasets via io/h5.py
    # -----------------------------
    with open_h5(h5_path, "r") as h5:
        schema_version = str(h5_read_scalar(h5, "/meta/schema_version"))
        run_id = str(h5_read_scalar(h5, "/meta/run_id"))

        # Device
        vessel = h5_read_array(h5, "/device/vessel_boundary") if _maybe(h5, "/device/vessel_boundary") else None

        coil_names = None
        if _maybe(h5, "/device/coils/names"):
            coil_names = _decode_strings(h5_read_array(h5, "/device/coils/names"))
        coil_centers = h5_read_array(h5, "/device/coils/centers") if _maybe(h5, "/device/coils/centers") else None
        coil_radii = h5_read_array(h5, "/device/coils/radii") if _maybe(h5, "/device/coils/radii") else None
        coil_I = h5_read_array(h5, "/device/coils/I_pf") if _maybe(h5, "/device/coils/I_pf") else None
        coil_Imax = h5_read_array(h5, "/device/coils/I_max") if _maybe(h5, "/device/coils/I_max") else None

        # Target
        target_boundary = h5_read_array(h5, "/target/boundary") if _maybe(h5, "/target/boundary") else None
        psi_target = None
        if _maybe(h5, "/target/psi_boundary"):
            psi_target = float(h5_read_scalar(h5, "/target/psi_boundary"))

        # Grid
        R = h5_read_array(h5, "/grid/R") if _maybe(h5, "/grid/R") else None
        Z = h5_read_array(h5, "/grid/Z") if _maybe(h5, "/grid/Z") else None
        RR = h5_read_array(h5, "/grid/RR") if _maybe(h5, "/grid/RR") else None

        # Greens
        G_psi = h5_read_array(h5, "/device/coil_greens/psi_per_amp") if _maybe(h5, "/device/coil_greens/psi_per_amp") else None

        # 04 fit results (optional)
        fit_boundary_pts = h5_read_array(h5, "/optimization/fit_results/boundary_points") if _maybe(h5, "/optimization/fit_results/boundary_points") else None
        psi_boundary_fit = h5_read_array(h5, "/optimization/fit_results/psi_boundary_fit") if _maybe(h5, "/optimization/fit_results/psi_boundary_fit") else None
        clamped = h5_read_array(h5, "/optimization/fit_results/clamped") if _maybe(h5, "/optimization/fit_results/clamped") else None
        contour_rms = float(h5_read_scalar(h5, "/optimization/fit_results/contour_rms")) if _maybe(h5, "/optimization/fit_results/contour_rms") else None
        psi_ref = float(h5_read_scalar(h5, "/optimization/fit_results/psi_ref")) if _maybe(h5, "/optimization/fit_results/psi_ref") else None

        # Equilibrium (optional depending on stage)
        psi = h5_read_array(h5, "/equilibrium/fixed/psi") if _maybe(h5, "/equilibrium/fixed/psi") else None
        psi_axis = float(h5_read_scalar(h5, "/equilibrium/fixed/psi_axis")) if _maybe(h5, "/equilibrium/fixed/psi_axis") else None
        psi_lcfs = float(h5_read_scalar(h5, "/equilibrium/fixed/psi_lcfs")) if _maybe(h5, "/equilibrium/fixed/psi_lcfs") else psi_target

        p_psi = h5_read_array(h5, "/equilibrium/fixed/p_psi") if _maybe(h5, "/equilibrium/fixed/p_psi") else None
        F_psi = h5_read_array(h5, "/equilibrium/fixed/F_psi") if _maybe(h5, "/equilibrium/fixed/F_psi") else None
        jphi = h5_read_array(h5, "/equilibrium/fixed/jphi") if _maybe(h5, "/equilibrium/fixed/jphi") else None
        plasma_mask = h5_read_array(h5, "/equilibrium/fixed/plasma_mask") if _maybe(h5, "/equilibrium/fixed/plasma_mask") else None

        BR = h5_read_array(h5, "/equilibrium/fixed/fields/BR") if _maybe(h5, "/equilibrium/fixed/fields/BR") else None
        BZ = h5_read_array(h5, "/equilibrium/fixed/fields/BZ") if _maybe(h5, "/equilibrium/fixed/fields/BZ") else None
        Bphi = h5_read_array(h5, "/equilibrium/fixed/fields/Bphi") if _maybe(h5, "/equilibrium/fixed/fields/Bphi") else None

        # Derived
        Ip = float(h5_read_scalar(h5, "/equilibrium/fixed/derived/Ip")) if _maybe(h5, "/equilibrium/fixed/derived/Ip") else None
        beta_p = float(h5_read_scalar(h5, "/equilibrium/fixed/derived/beta_p")) if _maybe(h5, "/equilibrium/fixed/derived/beta_p") else None
        li = float(h5_read_scalar(h5, "/equilibrium/fixed/derived/li")) if _maybe(h5, "/equilibrium/fixed/derived/li") else None
        kappa = float(h5_read_scalar(h5, "/equilibrium/fixed/derived/kappa")) if _maybe(h5, "/equilibrium/fixed/derived/kappa") else None
        delta = float(h5_read_scalar(h5, "/equilibrium/fixed/derived/delta")) if _maybe(h5, "/equilibrium/fixed/derived/delta") else None
        q_prof = h5_read_array(h5, "/equilibrium/fixed/derived/q_profile") if _maybe(h5, "/equilibrium/fixed/derived/q_profile") else None

    # Normalize mask to boolean if present
    if plasma_mask is not None:
        plasma_mask = np.asarray(plasma_mask).astype(bool)

    # Normalize clamped (stored as uint8 sometimes)
    if clamped is not None:
        clamped = np.asarray(clamped).reshape(-1).astype(bool)

    # -----------------------------
    # 01: device geometry
    # -----------------------------
    fig, _ = plot_device_geometry(
        run_id=run_id,
        schema_version=schema_version,
        vessel_boundary=vessel,
        coil_names=coil_names,
        coil_centers=coil_centers,
        coil_radii=coil_radii,
        coil_currents=coil_I,
    )
    save_figure(fig, d01 / "device_geometry", args.formats, dpi=args.dpi)
    plt.close(fig)

    # -----------------------------
    # 02: target overlay
    # -----------------------------
    fig, _ = plot_target_overlay(
        run_id=run_id,
        schema_version=schema_version,
        vessel_boundary=vessel,
        coil_names=coil_names,
        coil_centers=coil_centers,
        coil_radii=coil_radii,
        target_boundary=target_boundary,
        psi_lcfs=psi_lcfs,
        annotate_shape=True,
    )
    save_figure(fig, d02 / "target_overlay", args.formats, dpi=args.dpi)
    plt.close(fig)

    # -----------------------------
    # 03: coil greens (optional)
    # -----------------------------
    wrote_any = False
    if (G_psi is not None) and (R is not None) and (Z is not None) and (coil_names is not None):
        greens = plot_coil_greens(
            run_id=run_id,
            schema_version=schema_version,
            R=R,
            Z=Z,
            vessel_boundary=vessel,
            coil_names=coil_names,
            coil_centers=coil_centers,
            G_psi=G_psi,
            which=args.coils,
            max_plots=args.greens_max,
            n_levels=args.greens_levels,
        )
        for cname, fig, _ in greens:
            safe = cname.replace("/", "_")
            save_figure(fig, d03 / f"greens_{safe}", args.formats, dpi=args.dpi)
            plt.close(fig)
            wrote_any = True

    # -----------------------------
    # 04: coil-fit diagnostics + vacuum flux surfaces (optional)
    # -----------------------------
    if (fit_boundary_pts is not None) and (psi_boundary_fit is not None):
        fig, _ = plot_boundary_psi_vs_angle(
            run_id=run_id,
            schema_version=schema_version,
            boundary_pts=fit_boundary_pts,
            psi_boundary_fit=psi_boundary_fit,
            mean_line=True,
        )
        save_figure(fig, d04 / "boundary_psi_vs_theta", args.formats, dpi=args.dpi)
        plt.close(fig)

        fig, _ = plot_boundary_dpsi(
            run_id=run_id,
            schema_version=schema_version,
            boundary_pts=fit_boundary_pts,
            psi_boundary_fit=psi_boundary_fit,
            contour_rms=contour_rms,
        )
        save_figure(fig, d04 / "delta_psi_vs_theta", args.formats, dpi=args.dpi)
        plt.close(fig)

        if (coil_I is not None) and (coil_Imax is not None):
            fig, _ = plot_currents_vs_limits(
                run_id=run_id,
                schema_version=schema_version,
                coil_names=coil_names,
                I_pf=coil_I,
                I_max=coil_Imax,
                clamped=clamped,
            )
            save_figure(fig, d04 / "currents_vs_limits", args.formats, dpi=args.dpi)
            plt.close(fig)

    # Vacuum flux surfaces from coils (requires greens + currents + grid)
    if (G_psi is not None) and (coil_I is not None) and (R is not None) and (Z is not None):
        psi_vac = compute_psi_vacuum(G_psi, coil_I)

        # A meaningful reference level to highlight:
        psi_ref_level = None
        if psi_ref is not None and np.isfinite(psi_ref):
            psi_ref_level = float(psi_ref)
        elif psi_boundary_fit is not None:
            psi_ref_level = float(np.mean(np.asarray(psi_boundary_fit, float)))

        fig, _ = plot_vacuum_psi_map(
            run_id=run_id,
            schema_version=schema_version,
            R=R, Z=Z, psi_vac=psi_vac,
            vessel_boundary=vessel,
            target_boundary=target_boundary,
        )
        save_figure(fig, d04 / "vacuum_psi_map", args.formats, dpi=args.dpi)
        plt.close(fig)

        fig, _ = plot_vacuum_flux_surfaces_compare(
            run_id=run_id,
            schema_version=schema_version,
            R=R, Z=Z, psi_vac=psi_vac,
            vessel_boundary=vessel,
            target_boundary=target_boundary,
            psi_boundary_ref=psi_ref_level,
            psi_gs=psi,
            psi_gs_axis=psi_axis,
            psi_gs_lcfs=psi_lcfs,
        )
        save_figure(fig, d04 / "vacuum_flux_surfaces_compare", args.formats, dpi=args.dpi)
        plt.close(fig)

    # -----------------------------
    # 10/11/12/13: equilibrium / fields / profiles / derived (if equilibrium exists)
    # -----------------------------
    if (R is not None) and (Z is not None) and (psi is not None):
        fig, _ = plot_psi_map(
            run_id=run_id,
            schema_version=schema_version,
            R=R, Z=Z, psi=psi,
            psi_axis=psi_axis,
            psi_lcfs=psi_lcfs,
            vessel_boundary=vessel,
            lcfs_boundary=target_boundary,
        )
        save_figure(fig, d10 / "psi_map", args.formats, dpi=args.dpi)
        plt.close(fig)

        # Scalar maps
        if p_psi is not None:
            fig, _ = plot_scalar_map(
                run_id=run_id, schema_version=schema_version,
                title="Pressure p(R,Z)",
                R=R, Z=Z, field=p_psi, cbar_label="p [Pa]",
                vessel_boundary=vessel, lcfs_boundary=target_boundary,
                mask=plasma_mask,
            )
            save_figure(fig, d10 / "p_map", args.formats, dpi=args.dpi)
            plt.close(fig)

        if F_psi is not None:
            fig, _ = plot_scalar_map(
                run_id=run_id, schema_version=schema_version,
                title="Toroidal field function F(R,Z)",
                R=R, Z=Z, field=F_psi, cbar_label="F [T·m]",
                vessel_boundary=vessel, lcfs_boundary=target_boundary,
                mask=plasma_mask,
            )
            save_figure(fig, d10 / "F_map", args.formats, dpi=args.dpi)
            plt.close(fig)

        if jphi is not None:
            fig, _ = plot_scalar_map(
                run_id=run_id, schema_version=schema_version,
                title="Toroidal current density jφ(R,Z)",
                R=R, Z=Z, field=jphi, cbar_label="jφ [A/m²]",
                vessel_boundary=vessel, lcfs_boundary=target_boundary,
                mask=plasma_mask,
            )
            save_figure(fig, d10 / "jphi_map", args.formats, dpi=args.dpi)
            plt.close(fig)

        # Field magnitude maps
        if (BR is not None) and (BZ is not None):
            Bp = np.sqrt(BR * BR + BZ * BZ)
            fig, _ = plot_scalar_map(
                run_id=run_id, schema_version=schema_version,
                title="Poloidal field magnitude Bp(R,Z)",
                R=R, Z=Z, field=Bp, cbar_label="Bp [T]",
                vessel_boundary=vessel, lcfs_boundary=target_boundary,
                mask=None,
            )
            save_figure(fig, d11 / "Bp_map", args.formats, dpi=args.dpi)
            plt.close(fig)

        if (BR is not None) and (BZ is not None) and (Bphi is not None):
            Bmag = np.sqrt(BR * BR + BZ * BZ + Bphi * Bphi)
            fig, _ = plot_scalar_map(
                run_id=run_id, schema_version=schema_version,
                title="Total field magnitude |B|(R,Z)",
                R=R, Z=Z, field=Bmag, cbar_label="|B| [T]",
                vessel_boundary=vessel, lcfs_boundary=target_boundary,
                mask=None,
            )
            save_figure(fig, d11 / "Bmag_map", args.formats, dpi=args.dpi)
            plt.close(fig)

        # Midplane cuts
        fig, _ = plot_midplane_cuts(
            run_id=run_id, schema_version=schema_version,
            R=R, Z=Z,
            psi=psi, p=p_psi, jphi=jphi,
            BR=BR, BZ=BZ, Bphi=Bphi,
            z0=0.0,
        )
        save_figure(fig, d11 / "midplane_cuts", args.formats, dpi=args.dpi)
        plt.close(fig)

        # Profiles vs ψ̄
        if (psi_axis is not None) and (psi_lcfs is not None) and (plasma_mask is not None):
            denom = (psi_lcfs - psi_axis)
            if abs(denom) > 0:
                psin = (psi - psi_axis) / denom
                psin = np.clip(psin, 0.0, 1.0)

                Bp = None
                Bmag = None
                if (BR is not None) and (BZ is not None):
                    Bp = np.sqrt(BR * BR + BZ * BZ)
                if (BR is not None) and (BZ is not None) and (Bphi is not None):
                    Bmag = np.sqrt(BR * BR + BZ * BZ + Bphi * Bphi)

                fig, _ = plot_profiles_vs_psin(
                    run_id=run_id,
                    schema_version=schema_version,
                    psin=psin,
                    mask=plasma_mask,
                    p=p_psi,
                    F=F_psi,
                    jphi=jphi,
                    Bp=bp if (bp := Bp) is not None else None,
                    Bmag=bm if (bm := Bmag) is not None else None,
                    nbins=45,
                )
                save_figure(fig, d12 / "profiles_vs_psin", args.formats, dpi=args.dpi)
                plt.close(fig)

        # Derived summary + q-profile
        fig = plt.figure(figsize=(8.5, 5.0))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(f"Run: {run_id}   (schema {schema_version})\nDerived summary")

        lines = []
        if Ip is not None:
            lines.append(f"Ip = {Ip:.6g} A")
        if beta_p is not None:
            lines.append(f"beta_p = {beta_p:.6g}")
        if li is not None:
            lines.append(f"li = {li:.6g}")
        if kappa is not None:
            lines.append(f"kappa = {kappa:.6g}")
        if delta is not None:
            lines.append(f"delta = {delta:.6g}")

        ax.text(0.03, 0.92, "\n".join(lines) if lines else "(no derived scalars found)", va="top", fontsize=12)
        save_figure(fig, d13 / "derived_summary", args.formats, dpi=args.dpi)
        plt.close(fig)

        if q_prof is not None and np.asarray(q_prof).shape[0] > 0:
            q_prof = np.asarray(q_prof, float)
            fig = plt.figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            ax.set_title(f"Run: {run_id}   (schema {schema_version})\nq-profile")
            ax.plot(q_prof[:, 0], q_prof[:, 1], lw=2.2)
            ax.set_xlabel("ψ̄")
            ax.set_ylabel("q")
            ax.grid(True, alpha=0.25)
            save_figure(fig, d13 / "q_profile", args.formats, dpi=args.dpi)
            plt.close(fig)

        # Exemplary 3D field lines
        if args.fieldlines and (BR is not None) and (BZ is not None) and (Bphi is not None):
            seeds = []

            if plasma_mask is not None and psi is not None and psi_axis is not None and psi_lcfs is not None:
                denom = (psi_lcfs - psi_axis)
                if abs(denom) > 0:
                    psin = (psi - psi_axis) / denom
                    iz0 = int(np.argmin(np.abs(Z - 0.0)))

                    inside = plasma_mask[iz0, :].astype(bool)
                    ps_mid = psin[iz0, :]
                    ok = inside & np.isfinite(ps_mid)
                    idx = np.where(ok)[0]

                    if idx.size > 10:
                        ps_mid_in = ps_mid[idx]
                        # handle inverted convention if it happens
                        if ps_mid_in[0] > ps_mid_in[-1]:
                            ps_mid_in = 1.0 - ps_mid_in

                        target_psin = np.array([0.2, 0.4, 0.6, 0.8], dtype=float)
                        chosen_ir = []
                        for p0 in target_psin:
                            safe = (ps_mid_in > 0.05) & (ps_mid_in < 0.95)
                            if not np.any(safe):
                                safe = np.ones_like(ps_mid_in, dtype=bool)
                            j = int(np.argmin(np.abs(ps_mid_in[safe] - p0)))
                            ir = int(idx[np.where(safe)[0][j]])
                            chosen_ir.append(ir)

                        chosen_ir = sorted(set(chosen_ir))
                        min_sep = 3
                        filtered = []
                        for ir in chosen_ir:
                            if not filtered or (ir - filtered[-1] >= min_sep):
                                filtered.append(ir)

                        if len(filtered) < 3:
                            lo = int(idx.min() + 0.10 * (idx.max() - idx.min()))
                            hi = int(idx.min() + 0.90 * (idx.max() - idx.min()))
                            filtered = np.linspace(lo, hi, 4).astype(int).tolist()

                        seeds = [(float(R[ir]), float(Z[iz0])) for ir in filtered]

            if not seeds:
                seeds = [(float(R[len(R)//2]), 0.0)]

            fig = plot_fieldlines_3d(
                run_id=run_id, schema_version=schema_version,
                R=R, Z=Z, BR=BR, BZ=BZ, Bphi=Bphi,
                seeds=seeds,
                ds=float(args.fieldline_ds),
                nsteps=int(args.fieldline_steps),
            )
            save_figure(fig, d11 / "fieldlines_3d", args.formats, dpi=args.dpi)
            plt.close(fig)

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\nQuicklook plots saved to:")
    print(f"  {fig_dir}")
    print("Subdirectories:")
    print("  01_device, 02_target, 03_greens, 04_coil_fit, 10_equilibrium, 11_fields, 12_profiles, 13_derived")
    print("")


if __name__ == "__main__":
    main()