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
from typing import Any, Dict, Optional, Union

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
