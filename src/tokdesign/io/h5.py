"""
h5.py
=====

HDF5 I/O utilities for the tokamak design workflow.

Goals
-----
• Provide a small, consistent API to read/write data in results.h5
• Avoid duplicated boilerplate across scripts
• Enforce conventions:
    - all arrays are stored as NumPy-compatible datasets
    - scalars stored as 0-d datasets (or attributes when appropriate)
    - groups are created automatically
    - overwriting is explicit and safe

Design notes
------------
• This module is intentionally "low level":
  it does not know tokamak physics, only how to store things reliably.

• Preferred conventions:
  - arrays: datasets
  - small metadata dictionaries: group attributes (or datasets if non-scalar)
  - strings: stored as UTF-8 variable-length strings (h5py special dtype)

Dependencies
------------
• h5py
• numpy
"""


from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Mapping
from datetime import datetime

import numpy as np
import h5py


# ============================================================
# TYPES
# ============================================================

PathLike = Union[str, Path]
Attrs = Optional[Dict[str, Any]]


# ============================================================
# LOW-LEVEL HELPERS
# ============================================================

def h5_ensure_group(h5: h5py.File, path: str) -> h5py.Group:
    """
    Ensure that a group exists at `path`. Create intermediate groups as needed.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        Group path like "/device/coils" or "device/coils".

    Returns
    -------
    group : h5py.Group
        The group object.
    """
    path = _normalize_h5_path(path)

    if path == "/":
        return h5["/"]

    # Create nested groups one by one
    current = h5["/"]
    for part in path.strip("/").split("/"):
        if part not in current:
            current = current.create_group(part)
        else:
            current = current[part]
    return current


def _normalize_h5_path(path: str) -> str:
    """
    Normalize an HDF5 path:
    • ensure leading slash
    • collapse redundant slashes
    """
    if not path:
        raise ValueError("Empty HDF5 path is not allowed.")
    path = path.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    # collapse accidental double slashes
    while "//" in path:
        path = path.replace("//", "/")
    return path


def _split_parent(path: str) -> tuple[str, str]:
    """
    Split an HDF5 path into (parent_group_path, dataset_name).
    Example: "/equilibrium/psi" -> ("/equilibrium", "psi")
    """
    path = _normalize_h5_path(path)
    if path == "/":
        raise ValueError("Cannot split root path '/'.")
    parent, name = path.rsplit("/", 1)
    if parent == "":
        parent = "/"
    if name == "":
        raise ValueError(f"Invalid dataset path: '{path}'")
    return parent, name


def _set_attrs(obj: Union[h5py.Dataset, h5py.Group], attrs: Attrs) -> None:
    """
    Set attributes on a dataset or group.
    Only stores JSON-ish / HDF5-friendly values (numbers/strings/arrays).
    """
    if not attrs:
        return
    for k, v in attrs.items():
        # h5py can store scalars, strings, and numpy arrays as attrs.
        # For lists/tuples, convert to numpy array if numeric or to string.
        if isinstance(v, (list, tuple)):
            try:
                v_arr = np.asarray(v)
                obj.attrs[k] = v_arr
            except Exception:
                obj.attrs[k] = str(v)
        else:
            obj.attrs[k] = v


def _string_dtype():
    """Return h5py dtype for variable-length UTF-8 strings."""
    return h5py.string_dtype(encoding="utf-8")


# ============================================================
# PUBLIC WRITE FUNCTIONS
# ============================================================

def h5_write_array(
    h5: h5py.File,
    path: str,
    arr: np.ndarray,
    attrs: Attrs = None,
    overwrite: bool = True,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
) -> None:
    """
    Write a NumPy array dataset at `path`.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        Dataset path, e.g. "/grid/RR"
    arr : np.ndarray
        Data to store.
    attrs : dict or None
        Optional dataset attributes (e.g. {"units": "m"}).
    overwrite : bool
        If True, delete existing dataset and replace it.
    compression : str or None
        Compression algorithm; "gzip" is widely supported.
        Use None for no compression.
    compression_opts : int or None
        Compression level for gzip (1-9).

    Returns
    -------
    None
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    parent, name = _split_parent(path)
    grp = h5_ensure_group(h5, parent)

    if name in grp:
        if overwrite:
            del grp[name]
        else:
            raise FileExistsError(f"Dataset already exists at {path}")

    # HDF5 cannot store dtype=object arrays directly; guard early.
    if arr.dtype == object:
        raise TypeError(
            f"Cannot store object dtype array at {path}. "
            "Convert to numeric dtype or store as strings explicitly."
        )

    kwargs = {}
    if compression is not None and arr.size > 0:
        kwargs["compression"] = compression
        if compression == "gzip" and compression_opts is not None:
            kwargs["compression_opts"] = compression_opts

    dset = grp.create_dataset(name, data=arr, **kwargs)
    _set_attrs(dset, attrs)


def h5_write_scalar(
    h5: h5py.File,
    path: str,
    value: Union[int, float, str, np.number],
    attrs: Attrs = None,
    overwrite: bool = True,
) -> None:
    """
    Write a scalar dataset at `path`.

    Scalars are stored as 0-d datasets so they behave consistently with arrays
    and are easy to read back without guessing attribute vs dataset.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        Dataset path, e.g. "/meta/schema_version"
    value : int, float, str, np.number
        Scalar value to store.
    attrs : dict or None
        Optional attributes.
    overwrite : bool
        Overwrite existing dataset if present.

    Returns
    -------
    None
    """
    parent, name = _split_parent(path)
    grp = h5_ensure_group(h5, parent)

    if name in grp:
        if overwrite:
            del grp[name]
        else:
            raise FileExistsError(f"Dataset already exists at {path}")

    # Store strings with UTF-8 dtype
    if isinstance(value, str):
        dset = grp.create_dataset(name, data=value, dtype=_string_dtype())
    else:
        dset = grp.create_dataset(name, data=value)

    _set_attrs(dset, attrs)


def h5_write_strings(
    h5: h5py.File,
    path: str,
    strings: list[str],
    attrs: Attrs = None,
    overwrite: bool = True,
) -> None:
    """
    Write a 1D list of strings as a dataset (UTF-8).

    Useful for coil names, etc.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        Dataset path.
    strings : list[str]
        List of strings to store.
    attrs : dict or None
        Optional attrs.
    overwrite : bool
        Overwrite existing dataset if present.
    """
    arr = np.asarray(strings, dtype=object)  # prepare object array for strings
    parent, name = _split_parent(path)
    grp = h5_ensure_group(h5, parent)

    if name in grp:
        if overwrite:
            del grp[name]
        else:
            raise FileExistsError(f"Dataset already exists at {path}")

    dset = grp.create_dataset(name, data=arr, dtype=_string_dtype())
    _set_attrs(dset, attrs)


def h5_write_dict_as_attrs(
    h5: h5py.File,
    group_path: str,
    d: Dict[str, Any],
    overwrite: bool = True,
) -> None:
    """
    Write a small dictionary into group attributes.

    This is best for small metadata (numbers, short strings, small arrays).
    For large nested dicts, prefer storing as datasets/subgroups.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    group_path : str
        Group path where attributes are stored.
    d : dict
        Dictionary of key->value.
    overwrite : bool
        If False, will raise if attribute key already exists.

    Returns
    -------
    None
    """
    grp = h5_ensure_group(h5, group_path)
    for k, v in d.items():
        if (k in grp.attrs) and (not overwrite):
            raise FileExistsError(f"Attribute already exists: {group_path}::{k}")
        # use same helper conversion logic
        _set_attrs(grp, {k: v})


# ============================================================
# PUBLIC READ FUNCTIONS
# ============================================================

def h5_read_array(h5: h5py.File, path: str) -> np.ndarray:
    """
    Read a dataset at `path` as a NumPy array.

    If the dataset is a string dataset, returns a NumPy array of Python str.
    """
    path = _normalize_h5_path(path)
    if path not in h5:
        raise KeyError(f"Dataset not found: {path}")

    data = h5[path][...]

    # Case 1: scalar bytes -> decode to str then wrap as 0-d array
    if isinstance(data, (bytes, np.bytes_)):
        return np.asarray(data.decode("utf-8"))

    arr = np.asarray(data)

    # Case 2: array of bytes -> decode elementwise
    # Covers fixed-length ASCII (dtype.kind == 'S') and object arrays of bytes.
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    if arr.dtype == object:
        # Some h5py string datasets can come back as object arrays of bytes
        if arr.size > 0 and isinstance(arr.flat[0], (bytes, np.bytes_)):
            return np.vectorize(lambda x: x.decode("utf-8"))(arr)

    return arr


def h5_read_scalar(h5: h5py.File, path: str) -> Any:
    """
    Read a scalar dataset at `path`.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        Dataset path.

    Returns
    -------
    Any
        Python scalar (int/float/str) when possible.
    """
    path = _normalize_h5_path(path)
    if path not in h5:
        raise KeyError(f"Dataset not found: {path}")
    dset = h5[path]
    if dset.shape != ():
        raise ValueError(f"Dataset at {path} is not scalar (shape={dset.shape})")

    val = dset[()]
    # h5py returns bytes for variable-length strings sometimes
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    # numpy scalars -> python scalars
    if isinstance(val, np.generic):
        return val.item()
    return val


def h5_read_attrs(h5: h5py.File, path: str) -> Dict[str, Any]:
    """
    Read attributes from a dataset or group.

    Parameters
    ----------
    h5 : h5py.File
        Open HDF5 file handle.
    path : str
        Group or dataset path.

    Returns
    -------
    dict
        Attributes dictionary (with bytes decoded if possible).
    """
    path = _normalize_h5_path(path)
    if path not in h5:
        raise KeyError(f"Object not found: {path}")

    obj = h5[path]
    out: Dict[str, Any] = {}
    for k, v in obj.attrs.items():
        if isinstance(v, (bytes, np.bytes_)):
            out[k] = v.decode("utf-8")
        else:
            out[k] = v
    return out


# ============================================================
# FILE OPEN CONVENIENCE
# ============================================================

def open_h5(path: PathLike, mode: str = "r") -> h5py.File:
    """
    Convenience wrapper to open an HDF5 file with a normalized Path.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.
    mode : str
        h5py open mode ("r", "r+", "w", "a", ...)

    Returns
    -------
    h5py.File
    """
    p = Path(path).expanduser().resolve()
    return h5py.File(p, mode)


# ============================================================
# HISTORY
# ============================================================

def h5_make_history_event_id() -> str:
    """
    Create a sortable UTC event id.
    Example: '2026-01-16T210534Z'
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")


def h5_snapshot_paths(
    h5: h5py.File,
    *,
    stage: str,
    src_paths: list[str],
    event_id: str | None = None,
    attrs: Attrs = None,
    overwrite_event: bool = False,
) -> str:
    """
    Snapshot selected HDF5 paths into /history/<stage>/<event_id>/...

    Each src path is copied preserving its internal structure:
        src '/device/coils/I_pf' ->
        dst '/history/<stage>/<event_id>/device/coils/I_pf'

    Parameters
    ----------
    stage : str
        Stage label, e.g. '04_fit_pf_currents'
    src_paths : list[str]
        Paths (datasets or groups) to snapshot if they exist.
    event_id : str or None
        If None, generates one.
    attrs : dict
        Stored as attributes on the event group.
    overwrite_event : bool
        If True and the event group exists, it is deleted and recreated.

    Returns
    -------
    event_id : str
        The event id used.
    """
    stage = str(stage).strip()
    if not stage:
        raise ValueError("stage must be a non-empty string.")

    if event_id is None:
        event_id = h5_make_history_event_id()

    base = f"/history/{stage}/{event_id}"
    base = _normalize_h5_path(base)

    # Ensure /history exists
    h5_ensure_group(h5, "/history")

    # (Re)create event group
    if base in h5:
        if overwrite_event:
            del h5[base]
        else:
            raise FileExistsError(f"History event already exists: {base}")

    ev_grp = h5_ensure_group(h5, base)
    _set_attrs(ev_grp, attrs)

    for src in src_paths:
        src = _normalize_h5_path(src)
        if src not in h5:
            continue  # snapshot only what exists

        dst = f"{base}/{src.lstrip('/')}"
        parent, name = _split_parent(dst)
        grp = h5_ensure_group(h5, parent)

        # If something exists there (unlikely unless overwrite_event=False), remove it.
        if name in grp:
            del grp[name]

        # h5py copy works for both datasets and groups
        h5.copy(src, grp, name=name)

    return event_id



# ============================================================
# Writing and reading data from the config (.yaml) files to .h5
# ============================================================

def h5_write_yaml_tree(
    h5: h5py.File,
    base_path: str,
    node: Any,
    *,
    key_sanitize: bool = True,
    overwrite: bool = True,
    attrs: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Recursively write a YAML-shaped Python object (dict/list/scalars) into HDF5.

    Intended use:
      - Stage 00 writes each loaded YAML config to:
          /input/<config_stem>/...
        by calling:
          h5_write_yaml_tree(h5, f"/input/{stem}", cfg)

    Mapping rules
    -------------
    - dict:
        Create/ensure group at base_path.
        Recurse into each key as a subgroup or dataset under base_path/<key>.

    - list/tuple:
        * empty list -> string dataset with attrs yaml_type="list", empty=1
        * list of dicts -> group base_path and items under base_path/<index>
        * list of strings -> string dataset
        * list of numeric/bool (non-None) -> array dataset
        * mixed scalars / includes None -> string dataset of str() values
        * otherwise -> group with indexed children (base_path/<index>)

    - scalar:
        * None -> scalar "null" with attrs yaml_type="null", is_null=1
        * bool -> store as int (0/1) with yaml_type="bool"
        * int/float -> numeric scalar with yaml_type="number"
        * str -> string scalar with yaml_type="str"
        * other -> string repr() with yaml_type="repr"

    Parameters
    ----------
    h5:
      Open h5py.File handle.
    base_path:
      HDF5 destination root (e.g. "/input/equilibrium_space").
    node:
      YAML-shaped Python object (dict/list/scalar).
    key_sanitize:
      Replace '/' with '_' in dict keys so keys can be used in paths.
    overwrite:
      If True, allow overwriting datasets. Groups are always ensured.
    attrs:
      Optional attributes merged into datasets/groups written at base_path level
      for some node types. (Per-node attrs still added like yaml_type.)
    """
    base = "/" + str(base_path).strip("/")

    def _sanitize_key(k: Any) -> str:
        s = str(k)
        return s.replace("/", "_") if key_sanitize else s

    def _is_number(x: Any) -> bool:
        return isinstance(x, (int, float, np.number)) and not isinstance(x, bool)

    def _is_scalar(x: Any) -> bool:
        return isinstance(x, (str, bool, int, float, np.number)) or (x is None)

    def _merge_attrs(a: Optional[Mapping[str, Any]], b: Mapping[str, Any]) -> dict:
        out = dict(a) if a else {}
        out.update(dict(b))
        return out

    # dict
    if isinstance(node, dict):
        h5_ensure_group(h5, base)
        if attrs:
            # Attach attrs to the group itself (optional)
            try:
                g = h5[base]
                for k, v in attrs.items():
                    g.attrs[k] = v
            except Exception:
                pass
        for k, v in node.items():
            kk = _sanitize_key(k)
            h5_write_yaml_tree(
                h5,
                f"{base}/{kk}",
                v,
                key_sanitize=key_sanitize,
                overwrite=overwrite,
            )
        return

    # list/tuple
    if isinstance(node, (list, tuple)):
        if len(node) == 0:
            a = _merge_attrs(attrs, {"yaml_type": "list", "empty": 1})
            h5_write_strings(h5, base, [], attrs=a, overwrite=overwrite)
            return

        if all(isinstance(x, dict) for x in node):
            h5_ensure_group(h5, base)
            if attrs:
                try:
                    g = h5[base]
                    for k, v in attrs.items():
                        g.attrs[k] = v
                except Exception:
                    pass
            for i, item in enumerate(node):
                h5_write_yaml_tree(
                    h5,
                    f"{base}/{i}",
                    item,
                    key_sanitize=key_sanitize,
                    overwrite=overwrite,
                )
            return

        if all(isinstance(x, str) for x in node):
            a = _merge_attrs(attrs, {"yaml_type": "list[str]"})
            h5_write_strings(h5, base, list(node), attrs=a, overwrite=overwrite)
            return

        if all(_is_scalar(x) for x in node):
            # numeric/bool only and no None -> numeric array
            if all((x is not None) and (isinstance(x, (bool, int, float, np.number))) for x in node):
                arr = np.asarray(node)
                # if object dtype sneaks in, fall back to strings
                if arr.dtype == object:
                    a = _merge_attrs(attrs, {"yaml_type": "list[mixed_scalar_repr]"})
                    h5_write_strings(h5, base, [repr(x) for x in node], attrs=a, overwrite=overwrite)
                else:
                    a = _merge_attrs(attrs, {"yaml_type": "list[numeric_or_bool]"})
                    h5_write_array(h5, base, arr, attrs=a, overwrite=overwrite)
                return

            # mixed scalars or includes None -> strings
            a = _merge_attrs(attrs, {"yaml_type": "list[mixed_scalar_as_str]"})
            h5_write_strings(
                h5,
                base,
                [("null" if x is None else str(x)) for x in node],
                attrs=a,
                overwrite=overwrite,
            )
            return

        # mixed complex list -> indexed children
        h5_ensure_group(h5, base)
        if attrs:
            try:
                g = h5[base]
                for k, v in attrs.items():
                    g.attrs[k] = v
            except Exception:
                pass
        for i, item in enumerate(node):
            h5_write_yaml_tree(
                h5,
                f"{base}/{i}",
                item,
                key_sanitize=key_sanitize,
                overwrite=overwrite,
            )
        return

    # scalar handling
    if node is None:
        a = _merge_attrs(attrs, {"yaml_type": "null", "is_null": 1})
        h5_write_scalar(h5, base, "null", attrs=a, overwrite=overwrite)
        return

    if isinstance(node, str):
        a = _merge_attrs(attrs, {"yaml_type": "str"})
        h5_write_scalar(h5, base, node, attrs=a, overwrite=overwrite)
        return

    if isinstance(node, bool):
        a = _merge_attrs(attrs, {"yaml_type": "bool"})
        h5_write_scalar(h5, base, int(node), attrs=a, overwrite=overwrite)
        return

    if _is_number(node):
        a = _merge_attrs(attrs, {"yaml_type": "number"})
        h5_write_scalar(h5, base, node, attrs=a, overwrite=overwrite)
        return

    # fallback
    a = _merge_attrs(attrs, {"yaml_type": "repr"})
    h5_write_scalar(h5, base, repr(node), attrs=a, overwrite=overwrite)


def h5_read_yaml_tree(h5: h5py.File, base_path: str) -> Any:
    """
    Reconstruct a YAML-shaped Python object (dict/list/scalars) from an HDF5 subtree.

    Intended use:
      cfg = h5_read_yaml_tree(h5, "/input/device_space")
      # cfg is a dict equivalent to yaml.safe_load(open("device_space.yaml"))

    This expects the tree was written by h5_write_yaml_tree().

    Notes on round-trip fidelity
    ----------------------------
    - dicts are reconstructed from groups.
    - lists are reconstructed either from:
        * datasets with attrs yaml_type starting with "list"
        * OR groups whose keys are consecutive integers: "0","1","2",...
    - scalars are reconstructed from datasets:
        * yaml_type="null" + attr is_null=1 -> None
        * yaml_type="bool" -> bool (stored as 0/1)
        * yaml_type="number" -> int/float (best-effort)
        * yaml_type="str" -> str
        * yaml_type="repr" -> str (repr string; cannot safely eval)
    """
    base = "/" + str(base_path).strip("/")

    if base not in h5:
        raise KeyError(f"HDF5 path not found: {base}")

    obj = h5[base]

    # -------------------------
    # Helpers
    # -------------------------

    def _is_indexed_group(g: h5py.Group) -> bool:
        """True if all keys are digit strings and form 0..N-1 (or at least all digit)."""
        keys = list(g.keys())
        if not keys:
            return False
        if not all(k.isdigit() for k in keys):
            return False
        # prefer strict contiguous behavior if possible
        idx = sorted(int(k) for k in keys)
        return idx == list(range(len(idx)))

    def _read_dataset(ds: h5py.Dataset) -> Any:
        # attrs may be bytes-like in some cases
        yaml_type = ds.attrs.get("yaml_type", None)
        if isinstance(yaml_type, bytes):
            yaml_type = yaml_type.decode("utf-8", errors="replace")

        # Null sentinel
        if ds.attrs.get("is_null", 0) == 1 or yaml_type == "null":
            return None

        # Read the raw value
        val = ds[()]

        # Decode bytes to str for scalar strings
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode("utf-8", errors="replace")

        # Handle string datasets (including variable-length strings)
        if isinstance(val, np.ndarray) and val.dtype.kind in ("S", "U", "O"):
            # Might be list-of-strings or mixed scalar list as strings
            out = []
            for x in val.tolist():
                if isinstance(x, (bytes, np.bytes_)):
                    out.append(x.decode("utf-8", errors="replace"))
                else:
                    out.append(str(x) if x is not None else "null")
            # If this was a list-of-scalars-as-str, we keep it as list[str]
            return out

        # Bool scalars stored as 0/1
        if yaml_type == "bool":
            try:
                return bool(int(val))
            except Exception:
                return bool(val)
        
        if yaml_type == "number":
            # Convert numpy scalar to plain Python scalar
            if isinstance(val, np.generic):
                return val.item()
            return val


        # If it's a numpy scalar (int/float), convert to python
        if isinstance(val, np.generic):
            return val.item()

        # If it's a numpy array (numeric list), return list
        if isinstance(val, np.ndarray):
            return val.tolist()

        return val

    def _read_node(node: Union[h5py.Group, h5py.Dataset]) -> Any:
        if isinstance(node, h5py.Dataset):
            return _read_dataset(node)

        # Group: could be dict-like or list-like
        g: h5py.Group = node

        # If group is indexed 0..N-1 -> list
        if _is_indexed_group(g):
            n = len(g.keys())
            return [_read_node(g[str(i)]) for i in range(n)]

        # Else dict
        out: Dict[str, Any] = {}
        for k in g.keys():
            # Skip internal provenance keys if you want EXACT yaml.safe_load output
            # (You stored these at /input/<stem>/_source_filename etc.)
            # We'll keep them by default? No: you asked for same output as YAML.
            # So we skip keys that start with "_" by convention.
            if k.startswith("_"):
                continue
            out[k] = _read_node(g[k])
        return out

    return _read_node(obj)


def h5_read_input_config(h5: h5py.File, name: str) -> Dict[str, Any]:
    """
    Convenience wrapper for reading an input config stored under /input/<name>.

    Example:
      cfg = h5_read_input_config(h5, "device_space")

    Returns:
      dict equivalent to yaml.safe_load(<original_yaml_file>)
    """
    base = f"/input/{name}"
    data = h5_read_yaml_tree(h5, base)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at {base}, got {type(data)}")
    return data


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    test_path = Path("/tmp/tokamak_h5_test.h5")
    if test_path.exists():
        test_path.unlink()

    with open_h5(test_path, "w") as h5:
        # groups
        h5_ensure_group(h5, "/meta")
        h5_ensure_group(h5, "/grid")

        # scalars
        h5_write_scalar(h5, "/meta/schema_version", "0.1")
        h5_write_scalar(h5, "/meta/created_utc", "2026-01-15T12:00:00Z")

        # arrays
        R = np.linspace(1.0, 2.0, 5)
        psi = np.random.rand(4, 5)
        h5_write_array(h5, "/grid/R", R, attrs={"units": "m"})
        h5_write_array(h5, "/equilibrium/psi", psi, attrs={"units": "Wb/rad"})

        # strings
        h5_write_strings(h5, "/device/coils/names", ["PF1U", "PF1L", "PF2"])

        # dict as attrs
        h5_write_dict_as_attrs(h5, "/meta", {"note": "test", "nr": 5})

    with open_h5(test_path, "r") as h5:
        assert h5_read_scalar(h5, "/meta/schema_version") == "0.1"
        R2 = h5_read_array(h5, "/grid/R")
        print(R2)
        assert np.allclose(R2, R)
        names = h5["/device/coils/names"][...]
        print(h5_read_array(h5,"/device/coils/names"))
        # names may come back as bytes; this is fine as long as you decode in readers when needed

    print(f"h5.py self-test passed. Wrote: {test_path}")
