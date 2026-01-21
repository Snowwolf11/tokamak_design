"""
tokdesign.optimization.objectives
=================================

Objective evaluation for Stage 01 (fixed-boundary equilibrium optimization).

This module is designed to match equilibrium_optimization.yaml, specifically:

  cfg_opt["objective"]:
    enabled: true/false
    blocks:
      - name: ...
        enabled: true/false
        weight: ...
        terms:
          - name: ...
            enabled: true/false
            metric: "<metric_name>"
            transform: {kind: "log"}              # optional
            compute_override: { ... }             # optional (meta-metric)
            penalty:
              kind: "target" | "band" | "hinge_upper" | "hinge_lower" | "linear" | "none"
              target/min/max/scale ...
            weight: ...

Outputs:
  total (float)
  blocks dict[str,float]  (includes block weights)
  terms dict[str,float]   (includes BOTH block and term weights; sums to total)

Notes
-----
- Metrics are retrieved preferentially from eq.scalars[name] (fast path).
- If missing, we try a small set of built-in fallbacks (profile-based).
- If still missing, we optionally call tokdesign.physics.metrics.evaluate_metric (future registry).

Compute overrides:
- The YAML includes meta-metrics like "controls_distance_to_bounds" that are computed
  directly from the control vector x and its bounds (not from equilibrium).
- For those, evaluate_objective accepts an optional `context` dict with x information.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, Callable
import math
import numpy as np

from tokdesign.physics.metrics_registry import evaluate_metric

# =============================================================================
# Public API
# =============================================================================

def evaluate_objective(
    eq: Any,
    cfg_opt: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Evaluate objective f for one equilibrium.

    Parameters
    ----------
    eq:
      Equilibrium result (object or dict-like). Should expose eq.scalars and/or profiles.

    cfg_opt:
      The equilibrium_optimization.yaml mapping.

    context:
      Optional dict for meta-metrics (compute_override), e.g.
        {
          "x": np.ndarray (n_active,),
          "x_bounds_lo": np.ndarray (n_active,),
          "x_bounds_hi": np.ndarray (n_active,),
          "x_names": list[str],
          "controls": <ControlMapping or similar> (optional),
        }

    Returns
    -------
    total, blocks, terms
    """
    context = context or {}

    obj_cfg = cfg_opt.get("objective", {})
    if not isinstance(obj_cfg, dict):
        return 0.0, {}, {}

    if not bool(obj_cfg.get("enabled", True)):
        return 0.0, {}, {}

    blocks_cfg = obj_cfg.get("blocks", [])
    if not isinstance(blocks_cfg, list):
        blocks_cfg = []

    total = 0.0
    blocks_out: Dict[str, float] = {}
    terms_out: Dict[str, float] = {}
    
    for bi, block in enumerate(blocks_cfg):
        if not isinstance(block, dict):
            continue
        if not bool(block.get("enabled", True)):
            continue
        
        block_name = str(block.get("name", f"block_{bi}"))
        w_block = _as_float(block.get("weight", 1.0), default=1.0)

        terms_cfg = block.get("terms", [])
        if not isinstance(terms_cfg, list):
            terms_cfg = []
        
        block_sum_unweighted = 0.0

        for ti, term in enumerate(terms_cfg):
            if not isinstance(term, dict):
                continue
            if not bool(term.get("enabled", True)):
                continue
            
            term_name = str(term.get("name", f"{block_name}.term_{ti}"))
            w_term = _as_float(term.get("weight", 1.0), default=1.0)

            # -----------------------
            # Metric value
            # -----------------------
            metric_name = term.get("metric", "")
            metric_name = str(metric_name) if metric_name is not None else ""
            metric_value = _compute_term_metric(eq, metric_name, term, cfg_opt, context)
            
            # -----------------------
            # Transform (optional)
            # -----------------------
            metric_value = _apply_transform(metric_value, term.get("transform", None))
            if np.isnan(metric_value) or (not np.isfinite(metric_value)): print(f"Warning: Metric: {metric_name} has value nan or inf")
            # -----------------------
            # Penalty (YAML uses "kind")
            # -----------------------
            penalty_spec = term.get("penalty", {"kind": "none"})
            penalty_value = apply_penalty(metric_value, penalty_spec)

            # Block sum (before block weight) is sum(w_term * penalty)
            block_sum_unweighted += w_term * penalty_value

            # Total includes block weight
            contrib = w_block * w_term * penalty_value
            #print(f"contrib:{contrib}, penalty_val:{penalty_value}, wterm:{w_term}, wblock:{w_block}")
            total += contrib
            terms_out[term_name] = float(contrib)

        blocks_out[block_name] = float(w_block * block_sum_unweighted)
    
    return float(total), blocks_out, terms_out


# =============================================================================
# Term metric computation (supports compute_override)
# =============================================================================

def _compute_term_metric(
    eq: Any,
    metric_name: str,
    term_cfg: Dict[str, Any],
    cfg_opt: Dict[str, Any],
    context: Dict[str, Any],
) -> float:
    """
    Compute term metric value with support for compute_override.

    YAML pattern:
      metric: "controls_distance_to_bounds"
      compute_override:
        enabled: true
        compute: "distance_to_bounds"
        params: { normalize: "by_range" }

    If compute_override.enabled is true, we compute it from context (x + bounds).
    Otherwise we compute from eq metrics registry.
    """
    # Compute override
    co = term_cfg.get("compute_override", None)
    if isinstance(co, dict) and bool(co.get("enabled", False)):
        compute = str(co.get("compute", "")).strip()
        params = co.get("params", {}) if isinstance(co.get("params", {}), dict) else {}
        return float(_compute_override_metric(compute, params, context))

    # Normal metric
    if not metric_name:
        return 0.0

    # Allow metric registry definition to carry params; your YAML uses metrics registry too,
    # but objective terms often reference just "metric: name".
    # If needed later: pull default params from cfg_opt["metrics"][metric_name]["params"].
    params = {}
    mreg = cfg_opt.get("metrics", {})
    if isinstance(mreg, dict) and metric_name in mreg and isinstance(mreg[metric_name], dict):
        params = mreg[metric_name].get("params", {}) if isinstance(mreg[metric_name].get("params", {}), dict) else {}

    return float(get_metric(eq, metric_name, params=params))


def _compute_override_metric(compute: str, params: Dict[str, Any], context: Dict[str, Any]) -> float:
    """
    Compute meta-metrics that depend on x/bounds rather than equilibrium.

    Currently implemented (to match your YAML):
      compute == "distance_to_bounds"
        normalize: "by_range" | "none"
        returns sum_j ((x - mid)/range)^2   (dimensionless)
    """
    compute = compute.strip().lower()
    if compute == "distance_to_bounds":
        x = np.asarray(context.get("x", []), dtype=float).reshape(-1)
        lo = np.asarray(context.get("x_bounds_lo", []), dtype=float).reshape(-1)
        hi = np.asarray(context.get("x_bounds_hi", []), dtype=float).reshape(-1)
        if x.size == 0 or lo.size != x.size or hi.size != x.size:
            return float("nan")

        mid = 0.5 * (lo + hi)
        rng = np.maximum(hi - lo, 1e-12)

        normalize = str(params.get("normalize", "by_range")).strip().lower()
        if normalize in ("by_range", "range"):
            z = (x - mid) / rng
        else:
            z = (x - mid)

        return float(np.sum(z**2))

    # Unknown override metric
    return float("nan")


# =============================================================================
# Metric evaluation
# =============================================================================

def get_metric(eq: Any, name: str, params: Optional[Dict[str, Any]] = None) -> float:
    """
    Retrieve a scalar metric value.

    Priority:
      1) eq.scalars[name]
      2) built-in fallback metrics here (profile-based)
      3) tokdesign.physics.metrics.evaluate_metric (future)
    """
    params = params or {}
    name = str(name)

    # 1) fast path: eq.scalars
    scalars = _get_attr(eq, "scalars", default=None)
    if isinstance(scalars, dict) and name in scalars:
        try:
            return float(scalars[name])
        except Exception:
            return float("nan")

    # 2) built-in fallbacks
    fn = _BUILTIN_METRICS.get(name)
    if fn is not None:
        try:
            return float(fn(eq, params))
        except Exception:
            return float("nan")

    # 3) external registry (pretend it exists later)
    if evaluate_metric is not None:
        try:
            return float(evaluate_metric(eq=eq, name=name, params=params))
        except Exception:
            return float("nan")
    return float(nan)


def _get_profile(eq: Any, key: str) -> Optional[np.ndarray]:
    """Fetch a 1D profile array from eq, supporting both attribute and dict forms."""
    if hasattr(eq, key):
        a = getattr(eq, key)
        return None if a is None else np.asarray(a, dtype=float)

    if isinstance(eq, dict):
        prof = eq.get("profiles", {})
        if isinstance(prof, dict) and key in prof:
            return np.asarray(prof[key], dtype=float)
        if key in eq:
            return np.asarray(eq[key], dtype=float)

    return None


def _apply_band(y: np.ndarray, x: np.ndarray, band: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a radial band selection.

    band can be:
      - dict {rho_min, rho_max}
      - string referring to cfg bands (NOT used here; registry handles that later)
    """
    if not isinstance(band, dict):
        return y, x
    rho_min = band.get("rho_min", None)
    rho_max = band.get("rho_max", None)
    if rho_min is None or rho_max is None:
        return y, x
    rho_min = float(rho_min)
    rho_max = float(rho_max)
    if rho_max <= rho_min:
        return y, x
    m = (x >= rho_min) & (x <= rho_max)
    if np.sum(m) < 5:
        return y, x
    return y[m], x[m]


def _smoothness_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Integral of squared second derivative: ∫ (d²y/dx²)² dx."""
    if y.size < 5:
        return 0.0
    dy = np.gradient(y, x, edge_order=1)
    d2 = np.gradient(dy, x, edge_order=1)
    return float(np.trapz(d2**2, x))


# Common fallbacks you already reference in YAML
def _m_q0(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    return float(q[0]) if q is not None and q.size else float("nan")


def _m_q95(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    if q is None or q.size == 0:
        return float("nan")
    idx = int(0.95 * (len(q) - 1))
    return float(q[idx])


def _m_q_min(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    return float(np.min(q)) if q is not None and q.size else float("nan")


def _m_rho_qmin(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    rho = _get_profile(eq, "rho")
    if q is None or rho is None or q.size == 0 or rho.size != q.size:
        return float("nan")
    return float(rho[int(np.argmin(q))])


def _m_low_q_volume_fraction(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    if q is None or q.size == 0:
        return float("nan")
    thr = float(params.get("q_threshold", 2.0))
    return float(np.mean(q < thr))


def _m_q_monotonicity_violation(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    rho = _get_profile(eq, "rho")
    if q is None or rho is None or q.size == 0 or rho.size != q.size:
        return float("nan")
    band = params.get("band", None)
    q_use, rho_use = _apply_band(q, rho, band)
    dq = np.gradient(q_use, rho_use, edge_order=1)
    # Integral of max(0, -dq/drho) (matches YAML intent)
    return float(np.trapz(np.maximum(-dq, 0.0), rho_use))


def _m_q_smoothness(eq: Any, params: Dict[str, Any]) -> float:
    q = _get_profile(eq, "q")
    rho = _get_profile(eq, "rho")
    if q is None or rho is None or q.size == 0 or rho.size != q.size:
        return float("nan")
    band = params.get("band", None)
    q_use, rho_use = _apply_band(q, rho, band)
    return _smoothness_integral(q_use, rho_use)


def _m_shear_smoothness(eq: Any, params: Dict[str, Any]) -> float:
    s = _get_profile(eq, "s")
    rho = _get_profile(eq, "rho")
    if s is None or rho is None or s.size == 0 or rho.size != s.size:
        return float("nan")
    band = params.get("band", None)
    s_use, rho_use = _apply_band(s, rho, band)
    return _smoothness_integral(s_use, rho_use)


def _m_negative_shear_extent(eq: Any, params: Dict[str, Any]) -> float:
    s = _get_profile(eq, "s")
    rho = _get_profile(eq, "rho")
    if s is None or rho is None or s.size == 0 or rho.size != s.size:
        return float("nan")
    band = params.get("band", None)
    s_use, _ = _apply_band(s, rho, band)
    # Fraction of points with negative shear (cheap proxy)
    return float(np.mean(s_use < 0.0))


_BUILTIN_METRICS: Dict[str, Callable[[Any, Dict[str, Any]], float]] = {
    "q0": _m_q0,
    "q95": _m_q95,
    "q_min": _m_q_min,
    "rho_qmin": _m_rho_qmin,
    "low_q_volume_fraction": _m_low_q_volume_fraction,
    "q_monotonicity_violation": _m_q_monotonicity_violation,
    "q_smoothness": _m_q_smoothness,
    "shear_smoothness": _m_shear_smoothness,
    "negative_shear_extent": _m_negative_shear_extent,
}


# =============================================================================
# Transforms
# =============================================================================

def _apply_transform(y: float, spec: Any) -> float:
    """
    YAML:
      transform: { kind: "log" }

    Supported:
      - log: log(max(y, eps))
      - abs: abs(y)
      - square: y^2
      - none / missing: identity
    """
    if spec is None:
        return float(y)
    if isinstance(spec, str):
        kind = spec
        params = {}
    elif isinstance(spec, dict):
        kind = spec.get("kind", "none")
        params = spec
    else:
        return float(y)

    kind = str(kind).strip().lower()

    if not np.isfinite(y):
        return float("nan")

    if kind in ("none", "identity"):
        return float(y)

    if kind == "log":
        eps = float(params.get("eps", 1e-12))
        return float(math.log(max(float(y), eps)))

    if kind == "abs":
        return float(abs(float(y)))

    if kind in ("square", "pow2"):
        yy = float(y)
        return float(yy * yy)

    # Unknown transform: identity (safe)
    return float(y)


# =============================================================================
# Penalties (YAML uses "kind")
# =============================================================================

def apply_penalty(y: float, spec: Any) -> float:
    """
    YAML penalty spec:
      penalty:
        kind: "target" | "band" | "hinge_lower" | "hinge_upper" | "linear"
        target/min/max/scale

    Convention:
      - returns >= 0
      - NaN metric -> +inf penalty (push away)
    """
    if spec is None:
        return 0.0
    if isinstance(spec, str):
        spec = {"kind": spec}
    if not isinstance(spec, dict):
        return 0.0

    kind = str(spec.get("kind", "none")).strip().lower()

    if not np.isfinite(y):
        print("Warning: y not finite (in objectives.py apply_penalty())")
        return float("inf")

    scale = max(_as_float(spec.get("scale", 1.0), default=1.0), 1e-12)

    if kind in ("none", "off", "disabled"):
        return 0.0

    # "linear" in your YAML: penalty = metric / scale (clipped at >=0 for safety)
    if kind == "linear":
        return float(max(float(y), 0.0) / scale)

    if kind == "target":
        target = _as_float(spec.get("target", 0.0), default=0.0)
        return float(((float(y) - target) / scale) ** 2)

    if kind == "hinge_lower":
        y_min = _as_float(spec.get("min", 0.0), default=0.0)
        if y >= y_min:
            return 0.0
        return float(((y_min - float(y)) / scale) ** 2)

    if kind == "hinge_upper":
        y_max = _as_float(spec.get("max", 0.0), default=0.0)
        if y <= y_max:
            return 0.0
        return float(((float(y) - y_max) / scale) ** 2)

    if kind == "band":
        y_min = _as_float(spec.get("min", -float("inf")), default=-float("inf"))
        y_max = _as_float(spec.get("max", +float("inf")), default=+float("inf"))
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if y < y_min:
            return float(((y_min - float(y)) / scale) ** 2)
        if y > y_max:
            return float(((float(y) - y_max) / scale) ** 2)
        return 0.0

    # Unknown kinds -> safe no-penalty
    return 0.0


# =============================================================================
# Small utilities
# =============================================================================

def _as_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        if isinstance(x, np.generic):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default
