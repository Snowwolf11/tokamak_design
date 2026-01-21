# src/tokdesign/optimization/constraints.py
"""
tokdesign.optimization.constraints
=================================

Hard constraints for Stage 01 (equilibrium optimization).

This module reads the constraints section from equilibrium_optimization.yaml:

constraints:
  enabled: true
  policy:
    on_violation: "reject"   # reject | penalty
  list:
    - name: "q0_min"
      enabled: true
      metric: "q0"
      op: ">="
      value: 1.05
    - ...

and turns it into an evaluatable ConstraintSet.

Key convention (IMPORTANT)
--------------------------
We store *margins* with a single sign convention:

  margin >= 0  -> constraint satisfied
  margin <  0  -> violated

For a scalar metric m and a scalar limit L:

  op ">="  (m >= L):  margin = m - L
  op "<="  (m <= L):  margin = L - m

This matches the HDF5 schema intent:
  /stage01_fixed/trace/constraints/margins
  /stage01_fixed/trace/constraints/ok

Design notes
------------
- Constraints are "hard": they are intended for feasibility decisions.
- The YAML includes policy.on_violation = "reject" | "penalty".
  That policy is usually enforced in stage01_fixed.py (the evaluator),
  but we parse and expose it here for clarity.

- We assume a metric registry exists and provides:
      registry.get(metric_name) -> value
  where value is usually a Python float or numpy scalar.

- This module keeps *just* constraint logic:
    parse config -> evaluate -> margins/ok flags
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


Number = Union[int, float, np.number]


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class ConstraintSpec:
    """
    A single scalar constraint.

    Attributes
    ----------
    name:
      Unique name for the constraint (used for trace storage).
    metric:
      Metric name to request from the metric registry.
    op:
      One of: ">=", "<=", "=="
    value:
      Limit value (scalar).
    enabled:
      If False, this constraint is ignored.
    tol:
      Optional tolerance for "==" constraints:
        satisfied if abs(metric - value) <= tol
      (Not present in your current YAML, but harmless to support.)
    """
    name: str
    metric: str
    op: str
    value: float
    enabled: bool = True
    tol: Optional[float] = None


@dataclass
class ConstraintEval:
    """
    Result of evaluating a ConstraintSet at one candidate.
    """
    names: List[str]
    margins: np.ndarray         # shape (n_constraints,)
    ok: np.ndarray              # shape (n_constraints,) dtype=bool

    def as_dict(self) -> Dict[str, float]:
        return {n: float(m) for n, m in zip(self.names, self.margins)}


class ConstraintConfigError(ValueError):
    """Raised for invalid or inconsistent constraint configuration."""


# ============================================================
# ConstraintSet
# ============================================================

class ConstraintSet:
    """
    Collection of scalar constraints.

    The set is built from the YAML config. Only enabled constraints are kept.
    """

    def __init__(
        self,
        specs: Sequence[ConstraintSpec],
        *,
        on_violation: str = "reject",
        strict_metrics: bool = True,
    ) -> None:
        self.specs: List[ConstraintSpec] = list(specs)
        self.on_violation = str(on_violation).strip().lower()
        if self.on_violation not in {"reject", "penalty"}:
            raise ConstraintConfigError(
                f"constraints.policy.on_violation must be 'reject' or 'penalty', got: {on_violation!r}"
            )
        self.strict_metrics = bool(strict_metrics)

        # Precompute names for stable ordering (important for trace arrays)
        self._names = [s.name for s in self.specs]

    @property
    def names(self) -> List[str]:
        """Enabled constraint names in stable order."""
        return list(self._names)

    @property
    def n_constraints(self) -> int:
        return len(self.specs)

    def evaluate(self, metric_registry: Any) -> ConstraintEval:
        """
        Evaluate all constraints at a single candidate.

        Parameters
        ----------
        metric_registry:
          Must support registry.get(metric_name) -> scalar value.

        Returns
        -------
        ConstraintEval
          margins and ok arrays aligned with self.names.
        """
        n = self.n_constraints
        margins = np.zeros((n,), dtype=float)
        ok = np.zeros((n,), dtype=bool)

        for i, spec in enumerate(self.specs):
            try:
                mval = metric_registry.get(spec.metric)
            except Exception as e:
                if self.strict_metrics:
                    raise RuntimeError(
                        f"Constraint '{spec.name}' requires metric '{spec.metric}', "
                        f"but registry.get() failed: {e}"
                    ) from e
                # Non-strict mode: treat missing metric as violated with big negative margin
                margins[i] = -np.inf
                ok[i] = False
                continue

            # Cast to float safely (numpy scalar / python scalar)
            try:
                m = float(mval)
            except Exception as e:
                raise TypeError(
                    f"Constraint '{spec.name}': metric '{spec.metric}' returned non-scalar "
                    f"value {type(mval)} which cannot be converted to float."
                ) from e

            margins[i] = _margin(spec.op, m, spec.value, tol=spec.tol)
            ok[i] = bool(margins[i] >= 0.0)

        return ConstraintEval(names=self.names, margins=margins, ok=ok)

    def any_violated(self, metric_registry: Any) -> bool:
        """Convenience: True if any constraint margin < 0."""
        ev = self.evaluate(metric_registry)
        return bool(np.any(ev.margins < 0.0))

    def violated_names(self, metric_registry: Any) -> List[str]:
        """Convenience: list of violated constraint names."""
        ev = self.evaluate(metric_registry)
        return [n for n, m in zip(ev.names, ev.margins) if m < 0.0]


# ============================================================
# Parsing from YAML config
# ============================================================

def build_constraints(
    cfg_opt: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Optional[ConstraintSet]:
    """
    Build a ConstraintSet from equilibrium_optimization.yaml content.

    Parameters
    ----------
    cfg_opt:
      The dict loaded from /input/equilibrium_optimization (YAML).
    strict:
      If True, missing/invalid fields raise ConstraintConfigError.
      If False, we skip malformed entries where possible.

    Returns
    -------
    ConstraintSet or None
      None if constraints.enabled is False or no enabled constraints exist.
    """
    croot = (cfg_opt.get("constraints") or {})
    enabled = bool(croot.get("enabled", False))
    if not enabled:
        return None

    policy = (croot.get("policy") or {})
    on_violation = str(policy.get("on_violation", "reject")).strip().lower()

    raw_list = croot.get("list", None)
    if raw_list is None:
        if strict:
            raise ConstraintConfigError("constraints.enabled=true but constraints.list is missing")
        return None

    if not isinstance(raw_list, list):
        raise ConstraintConfigError(f"constraints.list must be a list, got {type(raw_list)}")

    specs: List[ConstraintSpec] = []
    seen: set[str] = set()

    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            if strict:
                raise ConstraintConfigError(f"constraints.list[{idx}] must be a mapping, got {type(item)}")
            continue

        item_enabled = bool(item.get("enabled", True))
        if not item_enabled:
            continue

        name = item.get("name", None)
        metric = item.get("metric", None)
        op = item.get("op", None)
        value = item.get("value", None)
        tol = item.get("tol", None)  # optional (not in your current YAML)

        # Validate required fields
        if not isinstance(name, str) or not name.strip():
            if strict:
                raise ConstraintConfigError(f"constraints.list[{idx}] missing/invalid 'name'")
            continue
        name = name.strip()

        if name in seen:
            raise ConstraintConfigError(f"Duplicate constraint name: {name!r}")
        seen.add(name)

        if not isinstance(metric, str) or not metric.strip():
            if strict:
                raise ConstraintConfigError(f"Constraint {name!r}: missing/invalid 'metric'")
            continue
        metric = metric.strip()

        if not isinstance(op, str) or op.strip() not in {">=", "<=", "=="}:
            if strict:
                raise ConstraintConfigError(
                    f"Constraint {name!r}: 'op' must be one of >=, <=, ==; got {op!r}"
                )
            continue
        op = op.strip()

        # value must be scalar-like
        try:
            v = float(value)
        except Exception as e:
            if strict:
                raise ConstraintConfigError(f"Constraint {name!r}: invalid 'value' {value!r}") from e
            continue

        t: Optional[float] = None
        if tol is not None:
            try:
                t = float(tol)
            except Exception as e:
                if strict:
                    raise ConstraintConfigError(f"Constraint {name!r}: invalid 'tol' {tol!r}") from e
                t = None

        specs.append(
            ConstraintSpec(
                name=name,
                metric=metric,
                op=op,
                value=v,
                enabled=True,
                tol=t,
            )
        )

    if not specs:
        return None

    return ConstraintSet(specs, on_violation=on_violation, strict_metrics=True)


# ============================================================
# Core math: margin computation
# ============================================================

def _margin(op: str, metric_value: float, limit_value: float, *, tol: Optional[float] = None) -> float:
    """
    Convert (metric_value op limit_value) into a "margin" where >=0 means OK.

    For >=:
      margin = metric - limit
    For <=:
      margin = limit - metric
    For ==:
      margin = tol - abs(metric - limit)   (if tol provided)
      else margin = -abs(metric - limit)   (always violated unless exact)
    """
    if op == ">=":
        return float(metric_value - limit_value)
    if op == "<=":
        return float(limit_value - metric_value)
    if op == "==":
        d = abs(metric_value - limit_value)
        if tol is None:
            # Without a tolerance, equality is basically useless in floating point.
            # We still implement a sane convention.
            return float(-d)
        return float(tol - d)

    # Should never happen due to validation.
    raise ConstraintConfigError(f"Unsupported op: {op!r}")
