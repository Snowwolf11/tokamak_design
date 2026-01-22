# src/tokdesign/stage_outputs/stage01_report.py
#!/usr/bin/env python3
"""
stage01_report.py
=================

Stage 01 PDF report generator that reads from results.h5 (HDF5 is the stable contract).

This module:
• takes (run_dir, h5_path)
• reads Stage 01 outputs from /stage01_fixed/...
• produces a PDF in the run directory + a small assets folder for plots

No local HDF5 helper implementations:
• all HDF5 read/write helpers must be imported from tokdesign.io.h5

Plotting is delegated to tokdesign.viz modules (pure plotting functions).

Output
------
<run_dir>/stage01_report.pdf
<run_dir>/stage01_report_assets/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import shutil

import numpy as np

from tokdesign.io.h5 import open_h5, h5_read_array, h5_read_scalar

# viz (pure plotting)
from tokdesign.viz.plot_equilibrium import plot_psi_map, plot_scalar_map
from tokdesign.viz.plot_profiles import plot_profiles_vs_psin
from tokdesign.viz.plot_metrics import plot_radial_profiles

# reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.pdfgen import canvas as _canvas

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STAGE_ROOT = "/stage01_fixed"


# -----------------------------------------------------------------------------
# Small formatting helpers (non-HDF5)
# -----------------------------------------------------------------------------

def _fmt_num(v: Any) -> str:
    try:
        if v is None:
            return "—"
        if isinstance(v, (bool, np.bool_)):
            return "True" if bool(v) else "False"
        if isinstance(v, (int, np.integer)):
            return f"{int(v)}"
        if isinstance(v, (float, np.floating)):
            x = float(v)
            if not np.isfinite(x):
                return "NaN"
            ax = abs(x)
            if ax != 0 and (ax < 1e-3 or ax >= 1e4):
                return f"{x:.3e}"
            return f"{x:.6g}"
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return _fmt_num(v.reshape(-1)[0])
            return f"array{tuple(v.shape)}"
        return str(v)
    except Exception:
        return str(v)


def _fmt_bool(v: Any) -> str:
    try:
        if v is None:
            return "—"
        if isinstance(v, (bool, np.bool_)):
            return "True" if bool(v) else "False"
        if isinstance(v, (int, np.integer, float, np.floating)):
            return "True" if float(v) != 0.0 else "False"
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return _fmt_bool(v.reshape(-1)[0])
            return f"any={bool(np.any(v))}, all={bool(np.all(v))}"
        return "True" if bool(v) else "False"
    except Exception:
        return "—"


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def _page_decor(c: _canvas.Canvas, doc: SimpleDocTemplate, title: str) -> None:
    c.saveState()
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 1.7 * cm, 1.2 * cm, f"Page {doc.page}")
    c.drawString(1.7 * cm, 1.2 * cm, title)
    c.restoreState()


def _save_fig(fig, outpath: Path) -> Path:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return outpath


def _patch_reportlab_md5_compat() -> None:
    """
    ReportLab sometimes calls md5(data, usedforsecurity=False).
    Some Python/OpenSSL builds (notably older + certain conda combos) don't accept
    that kwarg or a 2nd positional arg. Patch both call sites used by ReportLab.
    """
    try:
        import hashlib
        from reportlab.pdfbase import pdfdoc
        from reportlab.lib import utils as rl_utils

        real_md5 = hashlib.md5

        def md5_compat(data=b"", *args, **kwargs):
            # ReportLab may pass usedforsecurity as kwarg or as a 2nd positional arg.
            kwargs.pop("usedforsecurity", None)
            # Ignore any extra positional args (e.g. usedforsecurity=False)
            return real_md5(data)

        pdfdoc.md5 = md5_compat
        rl_utils.md5 = md5_compat
    except Exception:
        pass


# -----------------------------------------------------------------------------
# HDF5 read helpers usage (no local helper implementations)
# -----------------------------------------------------------------------------

def _exists(h5, path: str) -> bool:
    # simple/explicit existence check; not an "HDF5 helper function"
    try:
        return path in h5
    except Exception:
        return False


def _read_optional_scalar(h5, path: str) -> Any:
    if not _exists(h5, path):
        return None
    try:
        return h5_read_scalar(h5, path)
    except Exception:
        return None


def _read_optional_array(h5, path: str) -> Optional[np.ndarray]:
    if not _exists(h5, path):
        return None
    try:
        return np.asarray(h5_read_array(h5, path))
    except Exception:
        return None


def _read_optional_strings(h5, path: str) -> Optional[List[str]]:
    arr = _read_optional_array(h5, path)
    if arr is None:
        return None
    out: List[str] = []
    for x in np.ravel(arr):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode("utf-8", errors="replace"))
        else:
            out.append(str(x))
    return out


# -----------------------------------------------------------------------------
# Report content builders
# -----------------------------------------------------------------------------

def _build_run_summary(h5) -> Table:
    best_eval = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/eval_index")
    best_obj = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/objective_total")

    feas = _read_optional_array(h5, f"{STAGE_ROOT}/trace/feasible")
    fail_code = _read_optional_array(h5, f"{STAGE_ROOT}/trace/fail_code")
    fail_reason = _read_optional_strings(h5, f"{STAGE_ROOT}/trace/fail_reason")

    feas_at_best = None
    fail_code_at_best = None
    fail_reason_at_best = None
    if best_eval is not None:
        i = int(best_eval)
        if feas is not None and 0 <= i < feas.shape[0]:
            feas_at_best = feas[i]
        if fail_code is not None and 0 <= i < fail_code.shape[0]:
            fail_code_at_best = fail_code[i]
        if fail_reason is not None and 0 <= i < len(fail_reason):
            fail_reason_at_best = fail_reason[i]

    psi_axis = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/equilibrium/psi_axis")
    psi_lcfs = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/equilibrium/psi_lcfs")
    axis_R = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/equilibrium/axis_R")
    axis_Z = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/equilibrium/axis_Z")

    data = [
        ["field", "value"],
        ["best eval_index", _fmt_num(best_eval)],
        ["best objective_total", _fmt_num(best_obj)],
        ["feasible at best", _fmt_bool(feas_at_best)],
        ["fail_code at best", _fmt_num(fail_code_at_best)],
        ["fail_reason at best", _fmt_num(fail_reason_at_best)],
        ["psi_axis", _fmt_num(psi_axis)],
        ["psi_lcfs", _fmt_num(psi_lcfs)],
        ["axis (R,Z)", f"({_fmt_num(axis_R)}, {_fmt_num(axis_Z)})"],
    ]
    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[5.5 * cm, 8.0 * cm])
    t.setStyle(_table_style())
    return t


def _build_active_controls_table(h5) -> Optional[Table]:
    names = _read_optional_strings(h5, f"{STAGE_ROOT}/problem/active_controls/names")
    lo = _read_optional_array(h5, f"{STAGE_ROOT}/problem/active_controls/bounds_lo")
    hi = _read_optional_array(h5, f"{STAGE_ROOT}/problem/active_controls/bounds_hi")
    x0 = _read_optional_array(h5, f"{STAGE_ROOT}/problem/active_controls/x_init")
    xb = _read_optional_array(h5, f"{STAGE_ROOT}/best/x")

    if not names:
        return None

    n = len(names)

    def _at(v: Optional[np.ndarray], i: int) -> str:
        if v is None or i >= v.shape[0]:
            return "—"
        return _fmt_num(v[i])

    data = [["name", "lo", "hi", "x_init", "x_best"]]
    for i in range(n):
        data.append([names[i], _at(lo, i), _at(hi, i), _at(x0, i), _at(xb, i)])

    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(_table_style())
    return t


def _build_constraints_table(h5) -> Optional[Table]:
    names = _read_optional_strings(h5, f"{STAGE_ROOT}/trace/constraints/names")
    margins = _read_optional_array(h5, f"{STAGE_ROOT}/best/constraints_margins")
    if not names or margins is None:
        return None

    data = [["constraint", "margin", "ok (margin>=0)"]]
    for i, nm in enumerate(names):
        m = float(margins[i]) if i < margins.shape[0] and np.isfinite(margins[i]) else np.nan
        ok = bool(np.isfinite(m) and (m >= 0.0))
        data.append([nm, _fmt_num(m), "True" if ok else "False"])

    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(_table_style())
    return t


def _build_best_metrics_table(h5, max_rows: int = 60) -> Optional[Table]:
    base = f"{STAGE_ROOT}/best/metrics"
    if not _exists(h5, base):
        return None

    keys = []
    try:
        keys = sorted(list(h5[base].keys()))
    except Exception:
        keys = []

    if not keys:
        return None

    data = [["metric", "value"]]
    for k in keys[:max_rows]:
        v = _read_optional_scalar(h5, f"{base}/{k}")
        data.append([k, _fmt_num(v)])

    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[8.5 * cm, 4.0 * cm])
    t.setStyle(_table_style())
    return t


def _build_meta_table(h5, max_rows: int = 50) -> Optional[Table]:
    base = f"{STAGE_ROOT}/meta"
    if not _exists(h5, base):
        return None
    keys = []
    try:
        keys = sorted(list(h5[base].keys()))
    except Exception:
        keys = []
    if not keys:
        return None

    data = [["meta key", "value"]]
    for k in keys[:max_rows]:
        v = _read_optional_scalar(h5, f"{base}/{k}")
        data.append([k, _fmt_num(v)])

    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[6.0 * cm, 7.5 * cm])
    t.setStyle(_table_style())
    return t


def delete_dir_if_all_match(out: List[Path], expected_dir: Path) -> None:
    # Basic validation
    if not out:
        raise ValueError("Path list is empty.")

    # Resolve everything (avoids relative path issues)
    out = [p.resolve() for p in out]
    expected_dir = expected_dir.resolve()

    # Check all files exist
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Some files do not exist: {[str(p) for p in missing]}"
        )

    # Get parent directories
    parents = {p.parent for p in out}

    # Check they all share the same directory
    if len(parents) != 1:
        raise ValueError(
            f"Files are in different directories: {[str(p) for p in parents]}"
        )

    actual_dir = parents.pop()

    # Check directory matches expected
    if actual_dir != expected_dir:
        raise ValueError(
            f"Directory mismatch.\n"
            f"Expected: {expected_dir}\n"
            f"Found:    {actual_dir}"
        )

    # Safety check
    if not actual_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {actual_dir}")

    # Delete directory recursively
    shutil.rmtree(actual_dir)
    print(f"Deleted directory: {actual_dir}")


# -----------------------------------------------------------------------------
# Plot orchestration (HDF5 read here, plotting delegated to viz)
# -----------------------------------------------------------------------------

def _make_plots(run_id: str, schema_version: str, run_dir: Path, h5) -> List[Path]:
    assets = run_dir / "stage01_report_assets"
    assets.mkdir(parents=True, exist_ok=True)

    out: List[Path] = []

    # Grid
    R = _read_optional_array(h5, f"{STAGE_ROOT}/grid/R")
    Z = _read_optional_array(h5, f"{STAGE_ROOT}/grid/Z")
    if R is None or Z is None:
        return out

    # Equilibrium core
    psi = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/psi")
    psi_axis = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/equilibrium/psi_axis")
    psi_lcfs = _read_optional_scalar(h5, f"{STAGE_ROOT}/best/equilibrium/psi_lcfs")

    lcfs_R = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/lcfs/R")
    lcfs_Z = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/lcfs/Z")
    lcfs_poly = None
    if lcfs_R is not None and lcfs_Z is not None and lcfs_R.size > 3 and lcfs_Z.size > 3:
        lcfs_poly = np.column_stack([lcfs_R.ravel(), lcfs_Z.ravel()])

    # Optional vessel boundary (outside stage01, but often present in results.h5)
    vessel_poly = None
    vb = _read_optional_array(h5, "/device/vessel_boundary")
    if vb is None:
        vb = _read_optional_array(h5, "/device/vessel/boundary")
    if vb is not None and vb.ndim == 2 and vb.shape[1] == 2 and vb.shape[0] >= 3:
        vessel_poly = vb

    if psi is not None:
        fig, _ax = plot_psi_map(
            run_id=run_id,
            schema_version=schema_version,
            R=R,
            Z=Z,
            psi=psi,
            psi_axis=psi_axis,
            psi_lcfs=psi_lcfs,
            vessel_boundary=vessel_poly,
            lcfs_boundary=lcfs_poly,
        )
        out.append(_save_fig(fig, assets / "psi_map.png"))

    # j_phi map
    jphi = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/j_phi")
    if jphi is not None:
        fig, _ax = plot_scalar_map(
            run_id=run_id,
            schema_version=schema_version,
            title="Equilibrium: toroidal current density jφ(R,Z)",
            R=R,
            Z=Z,
            field=jphi,
            cbar_label="jφ [A/m²]",
            vessel_boundary=vessel_poly,
            lcfs_boundary=lcfs_poly,
        )
        out.append(_save_fig(fig, assets / "jphi_map.png"))

    # Profiles vs psin (binned from 2D, using mask)
    mask = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/plasma_mask")
    if mask is not None and psi is not None and (psi_axis is not None) and (psi_lcfs is not None):
        maskb = np.asarray(mask, bool)
        denom = float(psi_lcfs) - float(psi_axis)
        if abs(denom) > 0:
            psin = (np.asarray(psi, float) - float(psi_axis)) / denom
            F2 = None
            # Note: stage01_fixed stores 1D profiles for F, but not 2D F(R,Z).
            # So for binned-vs-psin, we use what we have from 2D: p/jphi/B fields.
            BR = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/fields/BR")
            BZ = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/fields/BZ")
            Bphi = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/fields/Bphi")
            Bp = None
            Bmag = None
            if BR is not None and BZ is not None:
                Bp = np.sqrt(np.asarray(BR, float) ** 2 + np.asarray(BZ, float) ** 2)
            if BR is not None and BZ is not None and Bphi is not None:
                Bmag = np.sqrt(np.asarray(BR, float) ** 2 + np.asarray(BZ, float) ** 2 + np.asarray(Bphi, float) ** 2)

            p2 = None  # not stored as 2D in stage01_fixed (only 1D best/profiles/p)
            # but if you *do* have it elsewhere someday, quicklook/report will pick it up here:
            p2 = _read_optional_array(h5, f"{STAGE_ROOT}/best/equilibrium/p")  # optional future path

            fig, _ax = plot_profiles_vs_psin(
                run_id=run_id,
                schema_version=schema_version,
                psin=psin,
                mask=maskb,
                p=p2,
                F=F2,
                jphi=jphi,
                Bp=bp if (bp := Bp) is not None else None,
                Bmag=bm if (bm := Bmag) is not None else None,
            )
            out.append(_save_fig(fig, assets / "profiles_binned_vs_psin.png"))

    # 1D profiles (best/profiles/*)
    rho = _read_optional_array(h5, f"{STAGE_ROOT}/best/profiles/rho")
    if rho is not None:
        p1 = _read_optional_array(h5, f"{STAGE_ROOT}/best/profiles/p")
        F1 = _read_optional_array(h5, f"{STAGE_ROOT}/best/profiles/F")
        q1 = _read_optional_array(h5, f"{STAGE_ROOT}/best/profiles/q")
        s1 = _read_optional_array(h5, f"{STAGE_ROOT}/best/profiles/s")
        a1 = _read_optional_array(h5, f"{STAGE_ROOT}/best/profiles/alpha")
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
        out.append(_save_fig(fig, assets / "profiles_1d_vs_rho.png"))

    # Traces
    obj = _read_optional_array(h5, f"{STAGE_ROOT}/trace/objective_total")
    feas = _read_optional_array(h5, f"{STAGE_ROOT}/trace/feasible")
    margins = _read_optional_array(h5, f"{STAGE_ROOT}/trace/constraints/margins")
    if obj is not None or feas is not None or margins is not None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.set_title(f"Run: {run_id}   (schema {schema_version})\nOptimization traces")
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
        out.append(_save_fig(fig, assets / "traces.png"))

    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def write_report(run_dir: str | Path, h5_path: str | Path) -> Path:
    run_dir = Path(run_dir)
    h5_path = Path(h5_path)

    pdf_path = run_dir / "stage01_report.pdf"

    _patch_reportlab_md5_compat()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], spaceAfter=10))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=11))

    title = "TokDesign – Stage 01 Report"
    subtitle = f"run_dir: {run_dir.name}<br/>h5: {h5_path.name}"

    story: List[Any] = []
    story.append(Paragraph(title, styles["H1x"]))
    story.append(Paragraph(subtitle, styles["Small"]))
    story.append(Spacer(1, 0.35 * cm))

    with open_h5(h5_path, "r") as h5:
        if STAGE_ROOT not in h5:
            story.append(Paragraph(f"<b>Error:</b> Missing group {STAGE_ROOT} in HDF5.", styles["BodyText"]))
        else:
            schema_version = str(_read_optional_scalar(h5, f"{STAGE_ROOT}/meta/schema_version") or
                                 _read_optional_scalar(h5, "/meta/schema_version") or
                                 "unknown")
            run_id = str(_read_optional_scalar(h5, "/meta/run_id") or run_dir.name)

            # Summary
            story.append(Paragraph("Run summary", styles["H2x"]))
            story.append(_build_run_summary(h5))
            story.append(Spacer(1, 0.35 * cm))

            meta = _build_meta_table(h5)
            if meta is not None:
                story.append(Paragraph("Meta", styles["H2x"]))
                story.append(meta)
                story.append(Spacer(1, 0.35 * cm))

            # Controls / constraints / metrics
            story.append(Paragraph("Active controls", styles["H2x"]))
            t = _build_active_controls_table(h5)
            story.append(t if t is not None else Paragraph("No active control data found.", styles["Small"]))
            story.append(Spacer(1, 0.35 * cm))

            story.append(Paragraph("Constraints at best", styles["H2x"]))
            t = _build_constraints_table(h5)
            story.append(t if t is not None else Paragraph("No constraint data found.", styles["Small"]))
            story.append(Spacer(1, 0.35 * cm))

            story.append(Paragraph("Best metrics (scalars)", styles["H2x"]))
            t = _build_best_metrics_table(h5)
            story.append(t if t is not None else Paragraph("No best metrics stored.", styles["Small"]))

            story.append(PageBreak())

            # Plots
            story.append(Paragraph("Plots", styles["H2x"]))
            plot_paths = _make_plots(run_id=run_id, schema_version=schema_version, run_dir=run_dir, h5=h5)
            if not plot_paths:
                story.append(Paragraph("No plot data available (missing equilibrium outputs).", styles["Small"]))
            else:
                for p in plot_paths:
                    story.append(Paragraph(p.stem.replace("_", " "), styles["Small"]))
                    story.append(Image(str(p), width=16.5 * cm, height=11.0 * cm))
                    story.append(Spacer(1, 0.35 * cm))
    

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.7 * cm,
        rightMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title=title,
        author="tokdesign",
    )

    def _decor_first(c: _canvas.Canvas, d: SimpleDocTemplate) -> None:
        _page_decor(c, d, title)

    def _decor_later(c: _canvas.Canvas, d: SimpleDocTemplate) -> None:
        _page_decor(c, d, title)

    doc.build(story, onFirstPage=_decor_first, onLaterPages=_decor_later)
    delete_dir_if_all_match(plot_paths, run_dir / "stage01_report_assets")
    return pdf_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None):
    import argparse

    ap = argparse.ArgumentParser(description="Generate Stage 01 PDF report from results.h5.")
    ap.add_argument("--run-dir", required=True, help="Run directory (where report is written).")
    ap.add_argument(
        "--h5",
        required=False,
        help="Path to results.h5. If omitted, assumes <run-dir>/results.h5",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    run_dir = Path(args.run_dir)
    h5_path = Path(args.h5) if args.h5 else (run_dir / "results.h5")
    pdf = write_report(run_dir, h5_path)
    print(f"[stage01_report] wrote {pdf}")


if __name__ == "__main__":
    main()
