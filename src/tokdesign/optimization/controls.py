# src/tokdesign/optimization/controls.py
"""
tokdesign.optimization.controls
===============================

Stage-01 control mapping: convert equilibrium_space.yaml into an "active control
vector" x and a deterministic mapping from x -> structured parameter dict.

Why this exists
---------------
Stage 01 optimizes an equilibrium in a user-defined "design space" given by
equilibrium_space.yaml. We want to:

  • keep YAML nice and hierarchical (groups like plasma_boundary, profiles, etc.)
  • build a flat optimization vector x only from entries that are active
  • still be able to reconstruct a structured parameter dict for physics solves

This module does not do physics. It only deals with parameter bookkeeping.

Expected YAML structure (equilibrium_space.yaml)
------------------------------------------------
cfg_space:
  conventions: ...
  models: ...
  linkages: ...
  variables:
    some_group:
      some_var:
        bounds: [lo, hi]
        init: ...
        scale: ...
        units: ...
        active: true/false
    ...

Leaf dicts are detected by presence of "bounds" and "init".

Output
------
build_control_mapping(cfg_space) returns a ControlMapping that provides:
  • x_names: list[str]
  • x_bounds_lo: np.ndarray (n,)
  • x_bounds_hi: np.ndarray (n,)
  • x_init: np.ndarray (n,)
  • x_to_params(x): returns dict with numeric controls inserted, plus conventions/models/linkages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np


# -----------------------------------------------------------------------------
# Public dataclass (matches what stage01_fixed expects)
# -----------------------------------------------------------------------------

@dataclass
class ControlMapping:
    """
    Flat control vector mapping for Stage 01.

    Notes
    -----
    - x is in *physical units* (same units as YAML bounds/init).
    - We keep the original hierarchical structure in params["controls"].
    """
    x_names: List[str]
    x_bounds_lo: np.ndarray
    x_bounds_hi: np.ndarray
    x_init: np.ndarray

    # Internal mapping: for each active variable index i, how to set it in the tree
    _active_paths: List[Tuple[str, ...]]          # path inside variables tree
    _all_values_init: Dict[Tuple[str, ...], float]  # init for all variables (active+inactive)
    _cfg_space: Dict[str, Any]

    def x_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Convert a flat x vector into a structured params dict for physics.

        Returned dict layout:
          params = {
            "conventions": <cfg_space.conventions>,
            "models": <cfg_space.models>,
            "linkages": <cfg_space.linkages>,
            "controls": <variables tree but leaves are numeric values>,
          }

        Linkages (derive_* rules) are applied after setting numeric controls.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != len(self._active_paths):
            raise ValueError(
                f"x has length {x.shape[0]} but expected {len(self._active_paths)} "
                f"(n_active={len(self._active_paths)})"
            )

        # Start from init values for ALL leaves (active and inactive)
        controls_tree = _build_controls_tree_from_inits(
            variables=self._cfg_space.get("variables", {}),
            all_init=self._all_values_init,
        )

        # Overwrite active leaves from x
        for i, path in enumerate(self._active_paths):
            _set_in_tree(controls_tree, list(path), float(x[i]))

        params: Dict[str, Any] = {
            "conventions": copy.deepcopy(self._cfg_space.get("conventions", {})),
            "models": copy.deepcopy(self._cfg_space.get("models", {})),
            "linkages": copy.deepcopy(self._cfg_space.get("linkages", {})),
            "controls": controls_tree,
        }

        # Apply linkages (e.g. derive F0 from B0*R0_B0_ref)
        _apply_linkages_inplace(params)

        return params

    def params_to_x(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Optional inverse mapping (not strictly required for Stage 01).

        Extracts active control values from params["controls"] using the same paths.
        """
        controls = params.get("controls", {})
        out = np.zeros((len(self._active_paths),), dtype=float)
        for i, path in enumerate(self._active_paths):
            out[i] = float(_get_from_tree(controls, list(path)))
        return out


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def build_control_mapping(cfg_space: Dict[str, Any]) -> ControlMapping:
    """
    Build a ControlMapping from equilibrium_space.yaml (already loaded via YAML/HDF5).

    Conventions
    -----------
    - Leaf variables are those dicts that contain 'bounds' and 'init'.
    - Active controls are leaves with active: true.
    - x_names use dot-notation inside variables, e.g.:
        "toroidal_field.B0"
        "plasma_boundary.R0"
        "profiles.pressure.alpha_p"

    Returns
    -------
    ControlMapping
    """
    if not isinstance(cfg_space, dict):
        raise TypeError("cfg_space must be a dict (YAML-loaded mapping).")

    variables = cfg_space.get("variables", {})
    if not isinstance(variables, dict):
        raise TypeError("cfg_space['variables'] must be a dict.")

    # Discover all leaf variable specs
    leaf_specs = _collect_leaf_specs(variables)

    if not leaf_specs:
        raise ValueError("No variable leaf specs found under cfg_space['variables'].")

    # Build init lookup for all leaves (active+inactive)
    all_init: Dict[Tuple[str, ...], float] = {}
    for path, spec in leaf_specs:
        all_init[path] = _coerce_float(spec.get("init", np.nan), name=f"{'.'.join(path)}.init")

    # Active leaves -> x vector
    active_paths: List[Tuple[str, ...]] = []
    x_names: List[str] = []
    x_lo: List[float] = []
    x_hi: List[float] = []
    x_init: List[float] = []

    for path, spec in leaf_specs:
        active = bool(spec.get("active", False))
        if not active:
            continue

        b = spec.get("bounds", None)
        if not (isinstance(b, (list, tuple)) and len(b) == 2):
            raise ValueError(f"Variable {'.'.join(path)} is active but has invalid bounds: {b}")

        lo = _coerce_float(b[0], name=f"{'.'.join(path)}.bounds[0]")
        hi = _coerce_float(b[1], name=f"{'.'.join(path)}.bounds[1]")

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Variable {'.'.join(path)} has invalid bounds: {b}")

        ini = _coerce_float(spec.get("init", np.nan), name=f"{'.'.join(path)}.init")

        # Ensure init is within bounds (clip rather than crash; keep dev-friendly)
        if ini < lo or ini > hi:
            ini_clipped = float(np.minimum(np.maximum(ini, lo), hi))
            ini = ini_clipped

        active_paths.append(path)
        x_names.append(".".join(path))
        x_lo.append(lo)
        x_hi.append(hi)
        x_init.append(ini)

    if not active_paths:
        raise ValueError("No active controls found (no variables with active: true).")

    return ControlMapping(
        x_names=x_names,
        x_bounds_lo=np.asarray(x_lo, dtype=float),
        x_bounds_hi=np.asarray(x_hi, dtype=float),
        x_init=np.asarray(x_init, dtype=float),
        _active_paths=active_paths,
        _all_values_init=all_init,
        _cfg_space=copy.deepcopy(cfg_space),
    )


# -----------------------------------------------------------------------------
# Leaf discovery
# -----------------------------------------------------------------------------

def _is_leaf_spec(node: Any) -> bool:
    """
    A variable leaf spec is a dict containing at least:
      - bounds
      - init
    """
    return isinstance(node, dict) and ("bounds" in node) and ("init" in node)


def _collect_leaf_specs(variables_tree: Dict[str, Any]) -> List[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    """
    Traverse cfg_space['variables'] and collect (path, spec_dict) for each leaf variable.

    Example output path:
      ('profiles', 'pressure', 'alpha_p')
    """
    out: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []

    def rec(node: Any, path: List[str]) -> None:
        if _is_leaf_spec(node):
            out.append((tuple(path), node))
            return
        if isinstance(node, dict):
            for k, v in node.items():
                rec(v, path + [str(k)])

    rec(variables_tree, [])
    return out


# -----------------------------------------------------------------------------
# Build controls tree (numeric values only) from init lookup
# -----------------------------------------------------------------------------

def _build_controls_tree_from_inits(
    variables: Dict[str, Any],
    all_init: Dict[Tuple[str, ...], float],
) -> Dict[str, Any]:
    """
    Recreate the variables tree, but replace each leaf spec dict with its numeric init value.
    """
    def rec(node: Any, path: List[str]) -> Any:
        if _is_leaf_spec(node):
            p = tuple(path)
            return float(all_init.get(p, np.nan))
        if isinstance(node, dict):
            return {str(k): rec(v, path + [str(k)]) for k, v in node.items()}
        # Shouldn't happen in your YAML, but keep safe:
        return copy.deepcopy(node)

    return rec(variables, [])


# -----------------------------------------------------------------------------
# Tree utilities
# -----------------------------------------------------------------------------

def _set_in_tree(tree: Dict[str, Any], path: List[str], value: float) -> None:
    """
    Set tree[path[0]][path[1]]...[path[-1]] = value, creating dicts as needed.
    """
    cur: Any = tree
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def _get_from_tree(tree: Dict[str, Any], path: List[str]) -> Any:
    """
    Get tree[path[0]][path[1]]...[path[-1]].
    Raises KeyError if missing.
    """
    cur: Any = tree
    for k in path:
        cur = cur[k]
    return cur


# -----------------------------------------------------------------------------
# Linkages
# -----------------------------------------------------------------------------

def _apply_linkages_inplace(params: Dict[str, Any]) -> None:
    """
    Apply cfg_space['linkages'] rules.

    Currently implemented:
      linkages.derive_F0_from_B0.enabled == true:
        F0 := B0 * R0_B0_ref
        stored into controls['profiles']['toroidal_field_function']['F0']

    This matches equilibrium_space.yaml:
      expression: "F0 = B0 * R0_B0_ref"
      applies_to: "profiles.toroidal_field_function.F0"

    Design note:
      Keep linkages here simple and explicit. If you later want a general expression
      system, implement it in a dedicated module and call it from here.
    """
    linkages = params.get("linkages", {})
    controls = params.get("controls", {})

    if not isinstance(linkages, dict) or not isinstance(controls, dict):
        return

    rule = linkages.get("derive_F0_from_B0", {})
    if not isinstance(rule, dict):
        return
    if not bool(rule.get("enabled", False)):
        return

    # Required inputs
    try:
        B0 = float(controls["toroidal_field"]["B0"])
        Rref = float(controls["toroidal_field"]["R0_B0_ref"])
    except Exception:
        # If these keys aren't present, silently skip (or raise later in validation)
        return

    F0 = B0 * Rref

    # Apply to the target path (as in YAML)
    try:
        controls["profiles"]["toroidal_field_function"]["F0"] = float(F0)
    except Exception:
        # Create missing branches if necessary
        if "profiles" not in controls or not isinstance(controls["profiles"], dict):
            controls["profiles"] = {}
        if "toroidal_field_function" not in controls["profiles"] or not isinstance(controls["profiles"]["toroidal_field_function"], dict):
            controls["profiles"]["toroidal_field_function"] = {}
        controls["profiles"]["toroidal_field_function"]["F0"] = float(F0)


# -----------------------------------------------------------------------------
# Numeric coercion helpers
# -----------------------------------------------------------------------------

def _coerce_float(x: Any, name: str = "") -> float:
    """
    Convert numeric-ish YAML values into float.

    Accepts ints, floats, numpy numbers, and numeric strings (rare but possible).
    Raises ValueError for non-numeric.
    """
    try:
        if isinstance(x, (np.generic,)):
            return float(x.item())
        return float(x)
    except Exception as e:
        raise ValueError(f"Could not convert {name}={x!r} to float") from e
