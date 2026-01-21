# src/tokdesign/physics/metrics_registry.py
"""
tokdesign.physics.metrics_registry
==================================

A lightweight *metric registry* for metrics that are:
  • not stored directly in eq.scalars, and/or
  • not covered by the simple "built-in" profile-derived fallbacks in
    tokdesign.optimization.objectives.get_metric().

This registry lets you implement advanced or niche metrics once, in a clean and
testable place, and have the objective/constraint system call them by name.

Usage
-----
In objective evaluation (or elsewhere):

    from tokdesign.physics.metrics_registry import evaluate_metric
    val = evaluate_metric(eq, "kappa_li_proxy", params={...})

To add a new metric:
  • implement a function fn(eq, params) -> float
  • register it via @register_metric("my_metric")

Design principles
-----------------
• Small, explicit, dependency-free.
• Metric functions should return float (or NaN if undefined).
• Metric functions should NOT throw for normal "not computable" situations.
  Instead return NaN and let upstream policy decide (skip term, penalty, reject, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np


MetricFn = Callable[[Any, Optional[Mapping[str, Any]]], float]


# -----------------------------------------------------------------------------
# Registry storage
# -----------------------------------------------------------------------------

_REGISTRY: Dict[str, MetricFn] = {}


class MetricNotRegistered(KeyError):
    """Raised when evaluate_metric() is called for a name that is not registered."""


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    """
    Decorator to register a metric function under a given name.

    Example
    -------
    @register_metric("my_metric")
    def my_metric(eq, params=None) -> float:
        ...
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Metric name must be a non-empty string.")
    key = name.strip()

    def _decorator(fn: MetricFn) -> MetricFn:
        if key in _REGISTRY:
            raise KeyError(f"Metric '{key}' already registered.")
        _REGISTRY[key] = fn
        return fn

    return _decorator


def is_registered(name: str) -> bool:
    """Return True if a metric name is registered."""
    return str(name).strip() in _REGISTRY


def list_metrics() -> list[str]:
    """List all registered metric names (sorted)."""
    return sorted(_REGISTRY.keys())


def evaluate_metric(eq: Any, name: str, *, params: Optional[Mapping[str, Any]] = None) -> float:
    """
    Evaluate a registered metric by name.

    Parameters
    ----------
    eq:
      Equilibrium result object. Expected to expose:
        - eq.scalars: dict[str, float]  (optional)
        - eq.profiles or profile attributes (optional):
            psi_bar, rho, q, s, alpha, p, F, ...
    name:
      Metric name, must be registered.
    params:
      Optional metric parameters from YAML registry.

    Returns
    -------
    float
      Metric value; may be NaN if not computable.

    Raises
    ------
    MetricNotRegistered
      If the metric is not registered.
    """
    key = str(name).strip()
    fn = _REGISTRY.get(key)
    if fn is None:
        raise MetricNotRegistered(f"Metric '{key}' not registered. Registered: {list_metrics()}")
    try:
        return float(fn(eq, params))
    except Exception:
        # Registry metrics should be robust; if they still throw, return NaN and let
        # upstream handle it (reject/penalty/skip).
        return float("nan")


# -----------------------------------------------------------------------------
# Small helpers (internal)
# -----------------------------------------------------------------------------

def _get_scalars(eq: Any) -> Dict[str, Any]:
    scalars = getattr(eq, "scalars", None)
    return scalars if isinstance(scalars, dict) else {}


def _get_profile(eq: Any, key: str) -> Optional[np.ndarray]:
    """
    Fetch a 1D profile array from eq, supporting both attribute and dict forms.

    Accepts:
      - eq.<key> attribute
      - eq.profiles dict (attribute or dict-like)
      - eq["profiles"][key] if eq itself is dict-like
    """
    # Attribute direct
    if hasattr(eq, key):
        v = getattr(eq, key)
        if v is None:
            return None
        return np.asarray(v, dtype=float)

    # eq.profiles attribute
    prof = getattr(eq, "profiles", None)
    if isinstance(prof, dict) and key in prof:
        return np.asarray(prof[key], dtype=float)

    # dict-like eq
    if isinstance(eq, dict):
        prof2 = eq.get("profiles", None)
        if isinstance(prof2, dict) and key in prof2:
            return np.asarray(prof2[key], dtype=float)
        if key in eq:
            return np.asarray(eq[key], dtype=float)

    return None


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Metrics you currently need (from your YAML)
# -----------------------------------------------------------------------------


@register_metric("kappa_li_proxy")
def metric_kappa_li_proxy(eq: Any, params: Optional[Mapping[str, Any]] = None) -> float:
    """
    Proxy metric combining elongation (kappa) and internal inductance (li).

    Motivation
    ----------
    Your objective currently references a "kappa_li_proxy" term. The intent is
    usually: avoid extreme shaping (kappa) at a given internal inductance (li),
    or reward efficient shaping.

    Definition (simple & robust)
    ----------------------------
    proxy = kappa / max(li, eps)

    where:
      kappa is taken from eq.scalars["kappa"] if available
      li is taken from eq.scalars["li"] if available

    Params (optional)
    -----------------
    eps: float (default 1e-6)
      Floor for li to avoid division by zero.
    """
    params = params or {}
    eps = float(params.get("eps", 1e-6))

    scal = _get_scalars(eq)
    kappa = scal.get("kappa", float("nan"))
    li = scal.get("li", float("nan"))

    if not (_finite(kappa) and _finite(li)):
        return float("nan")

    li = float(li)
    kappa = float(kappa)
    return float(kappa / max(li, eps))


@register_metric("s_alpha_envelope_margin_min")
def metric_s_alpha_envelope_margin_min(eq: Any, params: Optional[Mapping[str, Any]] = None) -> float:
    """
    Minimum margin of a simple s–alpha "envelope" proxy over a radial band.

    Motivation
    ----------
    Your YAML references 's_alpha_envelope_margin_min'. This is typically a proxy for
    local ballooning stability: you want 'shear' s to be sufficiently large compared
    to 'alpha' (pressure-gradient drive), i.e. s - C*alpha >= margin.

    Definition (simple & robust)
    ----------------------------
    Given profiles s(rho), alpha(rho):
      envelope(rho) = s(rho) - alpha_scale * alpha(rho)
      margin(rho)   = envelope(rho) - margin_offset
    Return:
      min_{rho in band} margin(rho)

    Params (optional)
    -----------------
    band: {rho_min: float, rho_max: float}
      If provided, compute min over that band. Otherwise use full profile.
    alpha_scale: float (default 1.0)
      Multiply alpha by this factor.
    margin_offset: float (default 0.0)
      Subtract offset before taking the min (lets you require strictly positive margin).
    min_points: int (default 5)
      If band selection yields fewer than min_points, fall back to full profile.
    """
    params = params or {}
    alpha_scale = float(params.get("alpha_scale", 1.0))
    margin_offset = float(params.get("margin_offset", 0.0))
    min_points = int(params.get("min_points", 5))

    s = _get_profile(eq, "s")
    a = _get_profile(eq, "alpha")
    rho = _get_profile(eq, "rho")

    if s is None or a is None:
        return float("nan")
    s = np.asarray(s, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)

    if s.size == 0 or a.size == 0 or s.size != a.size:
        return float("nan")

    # If rho is available and band is provided, select it
    y = s - alpha_scale * a - margin_offset

    band = params.get("band", None)
    if isinstance(band, dict) and rho is not None:
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size == y.size:
            rho_min = band.get("rho_min", None)
            rho_max = band.get("rho_max", None)
            if rho_min is not None and rho_max is not None:
                try:
                    rho_min = float(rho_min)
                    rho_max = float(rho_max)
                    if rho_max > rho_min:
                        m = (rho >= rho_min) & (rho <= rho_max)
                        if int(np.sum(m)) >= min_points:
                            y = y[m]
                except Exception:
                    # ignore band if malformed
                    pass

    if y.size == 0:
        return float("nan")

    return float(np.min(y))

@register_metric("s_alpha_envelope_negative_margin_integral")
def metric_s_alpha_envelope_negative_margin_integral(eq: Any, params: Optional[Mapping[str, Any]] = None) -> float:
    """
    Integral of the *negative part* of an s–alpha envelope margin proxy.

    Motivation
    ----------
    Your YAML references 's_alpha_envelope_negative_margin_integral'. This is a smooth
    measure of "how badly" the s–alpha proxy violates a desired margin over radius:
      - zero if margin >= 0 everywhere in the chosen band
      - positive if margin dips below 0, scaled by depth and width of the region

    Definition (proxy)
    ------------------
    margin(rho) = s(rho) - alpha_scale * alpha(rho) - margin_offset

    negative_part(rho) = max(0, -margin(rho))

    Return the integral over rho in the chosen band:
        ∫ negative_part(rho) d(rho)

    Params (optional)
    -----------------
    band: {rho_min: float, rho_max: float}
      Restrict the integral to a rho band if rho is available.
    alpha_scale: float (default 1.0)
    margin_offset: float (default 0.0)
    min_points: int (default 5)
      If band selection yields fewer than min_points, fall back to full profile.
    """
    params = params or {}
    alpha_scale = float(params.get("alpha_scale", 1.0))
    margin_offset = float(params.get("margin_offset", 0.0))
    min_points = int(params.get("min_points", 5))

    s = _get_profile(eq, "s")
    a = _get_profile(eq, "alpha")
    rho = _get_profile(eq, "rho")

    if s is None or a is None:
        return float("nan")

    s = np.asarray(s, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    if s.size == 0 or a.size == 0 or s.size != a.size:
        return float("nan")

    margin = s - alpha_scale * a - margin_offset
    neg = np.maximum(-margin, 0.0)

    # Apply optional rho band if rho is available
    if rho is not None:
        rho = np.asarray(rho, dtype=float).reshape(-1)
        if rho.size == neg.size:
            band = params.get("band", None)
            if isinstance(band, dict):
                rho_min = band.get("rho_min", None)
                rho_max = band.get("rho_max", None)
                if rho_min is not None and rho_max is not None:
                    try:
                        rho_min = float(rho_min)
                        rho_max = float(rho_max)
                        if rho_max > rho_min:
                            m = (rho >= rho_min) & (rho <= rho_max)
                            if int(np.sum(m)) >= min_points:
                                rho = rho[m]
                                neg = neg[m]
                    except Exception:
                        pass

            # If we still have at least 2 points, integrate with trapezoidal rule
            if neg.size >= 2:
                return float(np.trapz(neg, rho))

    # Fallback: no usable rho grid -> integrate with uniform spacing
    if neg.size == 0:
        return float("nan")
    if neg.size == 1:
        return float(neg[0])
    return float(np.trapz(neg, dx=1.0))