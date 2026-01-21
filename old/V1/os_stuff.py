import os
import re
import glob
import csv
import numpy as np
import h5py

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