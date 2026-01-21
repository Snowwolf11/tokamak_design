"""
vessel.py
=========

Vessel / wall geometry utilities.

Purpose
-------
Represent the vacuum vessel (or first wall) boundary in the R-Z plane as a 2D
closed polyline and provide common geometric queries used by the workflow:

• "Is this point inside the vessel?"
• "What is the minimum distance from a point to the vessel boundary?"
• "Generate a parametric vessel shape (ellipse) for early studies."

Conventions
-----------
• The vessel boundary is represented as an array of shape (N, 2):
    boundary[:, 0] = R  [m]
    boundary[:, 1] = Z  [m]

• We assume the vessel polyline is CLOSED for inside/outside tests.
  If the user provides an open polyline, we close it automatically.

Notes
-----
This is not a CAD system. It’s a lightweight 2D geometry module
sufficient for clearance constraints and plotting.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np


# ============================================================
# PUBLIC API
# ============================================================

def load_vessel_from_config(cfg: Dict[str, Any]) -> np.ndarray:
    """
    Load or generate a vessel boundary polyline from the device config.

    Expected config structure (baseline_device.yaml):
    -----------------------------------------------
    vessel:
      representation: "parametric" or "polyline"
      parametric:
        R_center: ...
        Z_center: ...
        a_wall: ...
        b_wall: ...
        n_points: ...
      polyline:
        points: [[R1, Z1], [R2, Z2], ...]

    Parameters
    ----------
    cfg : dict
        Parsed YAML dictionary for baseline_device.yaml

    Returns
    -------
    boundary : np.ndarray, shape (N,2)
        Closed vessel boundary polyline (last point equals first point).
    """
    vessel = cfg.get("vessel", {})
    rep = vessel.get("representation", "parametric").lower()

    if rep == "parametric":
        p = vessel.get("parametric", {})
        R0 = float(p["R_center"])
        Z0 = float(p.get("Z_center", 0.0))
        a = float(p["a_wall"])
        b = float(p["b_wall"])
        n = int(p.get("n_points", 400))
        boundary = ellipse_boundary(R0, Z0, a, b, n_points=n)

    elif rep == "polyline":
        pl = vessel.get("polyline", {})
        pts = pl.get("points", None)
        if pts is None or len(pts) < 3:
            raise ValueError("vessel.polyline.points must contain at least 3 points.")
        boundary = np.asarray(pts, dtype=float)
    else:
        raise ValueError(f"Unknown vessel.representation: {rep}")

    boundary = ensure_closed_polyline(boundary)
    _validate_polyline(boundary)
    return boundary


def ellipse_boundary(
    R_center: float,
    Z_center: float,
    a_wall: float,
    b_wall: float,
    *,
    n_points: int = 400,
) -> np.ndarray:
    """
    Generate a simple elliptical vessel boundary.

    Parameters
    ----------
    R_center, Z_center : float
        Center of ellipse [m]
    a_wall : float
        Horizontal semi-axis [m]
    b_wall : float
        Vertical semi-axis [m]
    n_points : int
        Number of points along the boundary

    Returns
    -------
    boundary : np.ndarray, shape (n_points+1, 2)
        Closed polyline of ellipse boundary.
    """
    if n_points < 16:
        raise ValueError("n_points should be >= 16 for a reasonable boundary.")
    if a_wall <= 0 or b_wall <= 0:
        raise ValueError("Ellipse semi-axes must be positive.")

    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    R = R_center + a_wall * np.cos(theta)
    Z = Z_center + b_wall * np.sin(theta)
    boundary = np.column_stack([R, Z])
    return ensure_closed_polyline(boundary)


def ensure_closed_polyline(poly: np.ndarray) -> np.ndarray:
    """
    Ensure the polyline is closed by appending the first point at the end if needed.
    """
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("Polyline must have shape (N,2).")
    if poly.shape[0] < 3:
        raise ValueError("Polyline must contain at least 3 points.")

    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def point_in_polygon(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Vectorized point-in-polygon test (ray casting) for 2D polygons.

    Parameters
    ----------
    poly : np.ndarray, shape (N,2)
        Closed polygon boundary.
    pts : np.ndarray, shape (M,2)
        Points to test.

    Returns
    -------
    inside : np.ndarray, shape (M,)
        Boolean array indicating whether each point lies inside the polygon.

    Notes
    -----
    • Uses a standard ray casting method.
    • Points exactly on the boundary may be classified as inside or outside
      depending on floating point quirks; for our use (mask building), that’s OK.
    """
    poly = ensure_closed_polyline(poly)
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (M,2).")

    x = pts[:, 0]
    y = pts[:, 1]

    x0 = poly[:-1, 0]
    y0 = poly[:-1, 1]
    x1 = poly[1:, 0]
    y1 = poly[1:, 1]

    inside = np.zeros(len(pts), dtype=bool)

    # For each edge, check if it crosses a horizontal ray to +infinity.
    # Condition for crossing:
    #   (y0 > y) != (y1 > y)  and  x < x_intersection
    # where x_intersection is x coordinate where the edge crosses y.
    for i in range(len(x0)):
        yi0, yi1 = y0[i], y1[i]
        xi0, xi1 = x0[i], x1[i]

        cond = (yi0 > y) != (yi1 > y)
        # Avoid division by zero: if yi1==yi0, cond is false anyway
        x_int = xi0 + (y - yi0) * (xi1 - xi0) / (yi1 - yi0 + 1e-300)
        inside ^= cond & (x < x_int)

    return inside


def min_distance_to_polyline(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Compute minimum Euclidean distance from each point to the polyline segments.

    Parameters
    ----------
    poly : np.ndarray, shape (N,2)
        Polyline points (closed or open). Closed is recommended for vessel.
    pts : np.ndarray, shape (M,2)
        Query points.

    Returns
    -------
    dmin : np.ndarray, shape (M,)
        Minimum distance from each point to any segment of the polyline.

    Notes
    -----
    This is used for:
    • LCFS-to-wall clearance checks
    • coil-to-wall clearance checks (if needed)

    Implementation:
    • For each segment p0->p1, project point onto segment and clamp to [0,1].
    • Compute distance to closest point on the segment.
    • Take min over segments.
    """
    poly = np.asarray(poly, dtype=float)
    pts = np.asarray(pts, dtype=float)

    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("poly must have shape (N,2).")
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (M,2).")
    if poly.shape[0] < 2:
        raise ValueError("poly must have at least 2 points.")

    # Ensure closed so last segment is included (if desired)
    poly = ensure_closed_polyline(poly)

    p0 = poly[:-1]  # (S,2)
    p1 = poly[1:]   # (S,2)
    v = p1 - p0     # segment vectors, (S,2)
    vv = np.sum(v * v, axis=1)  # (S,)

    dmin = np.full(pts.shape[0], np.inf, dtype=float)

    # Loop over segments (S) – vessel boundaries typically have O(100-500) segments,
    # so this is fast enough for typical uses.
    for i in range(p0.shape[0]):
        a = p0[i]
        b = p1[i]
        ab = v[i]
        ab2 = vv[i]

        # Project each point p onto segment:
        # t = dot(p-a, ab) / |ab|^2
        # clamp t in [0,1]
        ap = pts - a
        t = (ap @ ab) / (ab2 + 1e-300)
        t = np.clip(t, 0.0, 1.0)

        closest = a + t[:, None] * ab
        d = np.linalg.norm(pts - closest, axis=1)
        dmin = np.minimum(dmin, d)

    return dmin


# ============================================================
# INTERNAL VALIDATION
# ============================================================

def _validate_polyline(poly: np.ndarray) -> None:
    """Basic sanity checks for vessel boundary polyline."""
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("Boundary polyline must have shape (N,2).")
    if poly.shape[0] < 4:
        raise ValueError("Closed boundary should have at least 4 points (including closure).")

    # Check finite
    if not np.all(np.isfinite(poly)):
        raise ValueError("Boundary polyline contains non-finite values.")

    # Check R > 0
    if np.any(poly[:, 0] <= 0.0):
        raise ValueError("Boundary polyline contains R <= 0 values (not valid for tokamak R-Z).")


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":
    print("Testing vessel.py")

    # Create ellipse vessel
    boundary = ellipse_boundary(1.7, 0.0, 1.0, 1.4, n_points=200)
    assert boundary.shape[1] == 2
    assert np.allclose(boundary[0], boundary[-1])

    # Point-in-polygon tests
    pts = np.array([
        [1.7, 0.0],   # center (inside)
        [0.1, 0.0],   # likely outside (R small)
        [2.7, 0.0],   # on right side (near boundary/outside)
    ])
    inside = point_in_polygon(boundary, pts)
    print("inside:", inside)

    # Distance tests
    d = min_distance_to_polyline(boundary, pts)
    print("dist:", d)

    print("vessel.py self-test passed")
