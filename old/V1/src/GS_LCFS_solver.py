"""
Grad–Shafranov fixed-boundary equilibrium solver (masked Dirichlet)
==================================================================

This script solves the axisymmetric Grad–Shafranov (GS) equation on a rectangular
(R, Z) grid using a finite-difference discretization and a sparse linear solver.

Key idea (embedded boundary / mask):
------------------------------------
We prescribe a last closed flux surface (LCFS) boundary curve (typically Miller).
We then build a boolean mask `inside_mask` identifying points inside the LCFS.

We enforce a *Dirichlet* boundary condition ψ = ψ_b:
  • on the outer edges of the rectangular computational box, AND
  • on every grid point outside the LCFS mask.

This is implemented by replacing the PDE row at those points with an identity row:
  A[p,p] = 1   and RHS b[p] = ψ_b
so that ψ_p = ψ_b is imposed exactly at those nodes.

Nonlinearity:
-------------
The RHS depends on ψ through the flux functions p(ψ) and F(ψ), therefore the
equation is nonlinear. We solve it by Picard iteration:
  ψ^k  -> compute RHS(ψ^k) -> solve linear system for ψ^(k+1) -> under-relax.

Saving:
-------
After solving, the solution is saved to a new run folder:
  <BASE_OUTPUT_DIR>/simXXX/gs_solution.h5
where XXX increments automatically.

Dependencies:
-------------
numpy, matplotlib, scipy, h5py
"""

import os
from pathlib import Path  # filesystem paths
import h5py               # HDF5 I/O (standard scientific format)

import numpy as np
import matplotlib.pyplot as plt

# IMPORTANT: Name collision! Matplotlib also has "Path" for polygons.
# We alias it to MplPath so pathlib.Path remains available.
from matplotlib.path import Path as MplPath

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab

# NOTE: splprep/splev are imported but not used below (left over from spline attempts).
# You can remove these imports safely unless you plan to use spline boundaries.
from scipy.interpolate import splprep, splev

# Vacuum permeability [H/m]
MU0 = 4e-7 * np.pi


# =============================================================================
# Optional: diverted boundary helper (NOT used if you stick with Miller LCFS)
# =============================================================================
def diverted_lcfs_polygon(
    R0=1.7, a=0.5, kappa=1.7, delta=0.33,
    Rx=None, Zx=None,
    leg_spread=0.35,     # leg separation scale (fraction of a)
    n_top=800,
    n_leg=300
):
    """
    Build a simple lower-single-null (LSN) *geometric* separatrix as a CLOSED POLYGON.

    This is NOT physics-based (no free-boundary / coils / vacuum). It is only used to
    create a polygon boundary for the mask approach.

    Construction (counter-clockwise):
      1) Top arc: Miller-like from outboard midplane -> top -> inboard midplane.
      2) Left "leg": inboard midplane down to X-point.
      3) Right "leg": X-point back up to outboard midplane.
    The legs are polylines (piecewise linear). This creates a cusp-like corner at the X-point.

    Returns
    -------
    Rb, Zb : 1D arrays
        Closed polygon coordinates. First point is repeated at the end.
    """
    # Default X-point guess if not provided
    if Rx is None:
        Rx = R0 + 0.15 * a         # outward X-point shift (typical)
    if Zx is None:
        Zx = -0.95 * kappa * a     # lower X-point

    # --- Top half (Z >= 0): Miller-ish arc from θ=0..π
    # θ=0 corresponds to outboard midplane (R=R0+a, Z=0).
    th = np.linspace(0.0, np.pi, n_top, endpoint=True)
    R_top = R0 + a * np.cos(th + delta * np.sin(th))
    Z_top = kappa * a * np.sin(th)

    # Convenience: midplane endpoints
    Rout_mid = (R0 + a, 0.0)
    Rin_mid  = (R0 - a, 0.0)

    # --- Define where legs "fan out" before heading to X-point (heuristic)
    # Leg anchor points spread around Rx.
    R_left_leg  = Rx - leg_spread * a
    R_right_leg = Rx + leg_spread * a
    Z_leg_knee  = -0.55 * kappa * a  # "knee" height above X-point (heuristic)

    # Left leg polyline: inboard midplane -> lower inboard -> leg anchor -> X-point
    R_leg_left = np.array([Rin_mid[0],   R0 - 0.75*a,  R_left_leg,  Rx])
    Z_leg_left = np.array([0.0,          Z_leg_knee,   Zx*0.85,     Zx])

    # Right leg polyline: X-point -> leg anchor -> lower outboard -> outboard midplane
    R_leg_right = np.array([Rx,          R_right_leg,  R0 + 0.75*a, Rout_mid[0]])
    Z_leg_right = np.array([Zx,          Zx*0.85,      Z_leg_knee,  0.0])

    def densify_polyline(Rp, Zp, n):
        """
        Densify a polyline by interpolating linearly in chord-length parameter.
        This produces more uniform point spacing along the polyline than interpolating
        simply by index.
        """
        t = np.linspace(0.0, 1.0, n)

        # Cumulative chord length s[k]
        s = np.zeros(len(Rp))
        s[1:] = np.cumsum(np.hypot(np.diff(Rp), np.diff(Zp)))
        if s[-1] == 0:
            return np.full(n, Rp[0]), np.full(n, Zp[0])
        s /= s[-1]

        # Interpolate coordinates against normalized chord length
        Rn = np.interp(t, s, Rp)
        Zn = np.interp(t, s, Zp)
        return Rn, Zn

    # Densify legs to avoid long straight segments when building the mask
    Rl, Zl = densify_polyline(R_leg_left,  Z_leg_left,  n_leg)
    Rr, Zr = densify_polyline(R_leg_right, Z_leg_right, n_leg)

    # Assemble closed polygon (CCW):
    # top arc starts at outboard midplane already. Avoid duplicating endpoints by slicing.
    Rb = np.concatenate([R_top, Rl[1:], Rr[1:]])
    Zb = np.concatenate([Z_top, Zl[1:], Zr[1:]])

    # Close explicitly (repeat the first point at the end)
    Rb = np.append(Rb, Rb[0])
    Zb = np.append(Zb, Zb[0])
    return Rb, Zb


# =============================================================================
# Boundary definition: Miller LCFS (your preferred simple fixed boundary)
# =============================================================================
def miller_lcfs(R0=1.7, a=0.8, kappa=1.6, delta=0.5, ntheta=2000):
    """
    Miller-like parametric boundary curve (limited configuration, no X-point).

    Parameterization used:
      R(θ) = R0 + a cos(θ + δ sinθ)
      Z(θ) = κ a sinθ

    Returns
    -------
    Rb, Zb : 1D arrays
        Boundary polygon points (not explicitly closed; that's okay for Path).
    """
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    Rb = R0 + a * np.cos(theta + delta * np.sin(theta))
    Zb = kappa * a * np.sin(theta)
    return Rb, Zb


def build_inside_mask(R, Z, Rb, Zb):
    """
    Create a boolean mask of points inside a boundary polygon.

    Parameters
    ----------
    R, Z : 1D arrays defining the rectangular grid
    Rb, Zb : 1D arrays describing the boundary polygon

    Returns
    -------
    inside : (NR,NZ) bool array
        True for grid points inside the polygon.
    """
    # Construct polygon vertices in (R,Z)
    poly = np.vstack([Rb, Zb]).T

    # MplPath is a point-in-polygon tester
    path = MplPath(poly, closed=True)

    # Create all grid points as list of (R,Z) and test containment
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    pts = np.vstack([RR.ravel(), ZZ.ravel()]).T
    inside = path.contains_points(pts).reshape(RR.shape)
    return inside


# =============================================================================
# Discretization: build sparse Δ* operator with embedded Dirichlet nodes
# =============================================================================
def build_delta_star_matrix_masked(R, Z, inside_mask):
    """
    Build sparse matrix A approximating Δ*ψ on the (R,Z) grid.

    Dirichlet nodes:
      • all points outside LCFS (inside_mask == False)
      • outer rectangular grid boundary (box edges)

    Implementation:
      For Dirichlet nodes we replace the PDE row with:
        A[p,p] = 1
      so the linear system enforces:
        ψ_p = b_p
      when we set RHS b_p = ψ_b at those nodes.

    Returns
    -------
    A : CSR sparse matrix (N x N)
    dR, dZ : grid spacings
    """
    NR = len(R)
    NZ = len(Z)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    N = NR * NZ
    A = lil_matrix((N, N), dtype=np.float64)  # LIL is convenient for incremental assembly

    def idx(i, j):
        """Flatten 2D index (i,j) -> 1D index p."""
        return i * NZ + j

    for i in range(NR):
        Ri = R[i]  # used in the -(1/R)∂ψ/∂R term
        for j in range(NZ):
            p = idx(i, j)

            # Identify Dirichlet nodes
            is_box_boundary = (i == 0) or (i == NR - 1) or (j == 0) or (j == NZ - 1)
            is_outside_lcfs = not inside_mask[i, j]

            if is_box_boundary or is_outside_lcfs:
                # Dirichlet: overwrite PDE with identity row => ψ_p = b_p
                A[p, p] = 1.0
                continue

            # Interior node stencil for:
            #   Δ*ψ = ∂²ψ/∂R² - (1/R)∂ψ/∂R + ∂²ψ/∂Z²
            # using 2nd-order central differences.
            c0  = -2.0 / dR**2 - 2.0 / dZ**2
            cRp =  1.0 / dR**2 - 1.0 / (2.0 * dR * Ri)
            cRm =  1.0 / dR**2 + 1.0 / (2.0 * dR * Ri)
            cZp =  1.0 / dZ**2
            cZm =  1.0 / dZ**2

            A[p, idx(i,   j)]   = c0
            A[p, idx(i+1, j)]   = cRp
            A[p, idx(i-1, j)]   = cRm
            A[p, idx(i,   j+1)] = cZp
            A[p, idx(i,   j-1)] = cZm

    # Convert to CSR for fast matrix-vector products in iterative solvers
    return A.tocsr(), dR, dZ


# =============================================================================
# Magnetic axis finder (sub-grid) + normalized flux ψ̄
# =============================================================================
def find_magnetic_axis_subgrid(R, Z, psi, inside_mask):
    """
    Estimate magnetic axis position (Rax, Zax) and ψ_axis.

    Method:
      1) Find the minimum ψ inside the plasma mask (grid point).
      2) Fit a 2D quadratic surface to a 3×3 stencil around that point:
           f(x,z) = a x^2 + b z^2 + c x z + d x + e z + g
         where x and z are local coordinates relative to the stencil center.
      3) Solve ∂f/∂x = 0, ∂f/∂z = 0 for the stationary point (sub-grid axis).
      4) Evaluate f at that point -> ψ_axis.

    This is cheap and avoids the axis being locked to a grid node.

    Returns
    -------
    Rax, Zax, psi_axis : floats
    """
    psi_inside = np.where(inside_mask, psi, np.inf)
    i0, j0 = np.unravel_index(np.argmin(psi_inside), psi.shape)

    # If too close to edges for a 3×3 stencil, fall back to the grid point
    if i0 < 1 or i0 > len(R) - 2 or j0 < 1 or j0 > len(Z) - 2:
        return R[i0], Z[j0], psi[i0, j0]

    x0 = R[i0]
    z0 = Z[j0]

    # Build least-squares system from 3×3 stencil
    rows = []
    vals = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii = i0 + di
            jj = j0 + dj
            if not inside_mask[ii, jj]:
                # Stencil crosses outside region -> fallback
                return R[i0], Z[j0], psi[i0, j0]
            x = R[ii] - x0
            z = Z[jj] - z0
            rows.append([x*x, z*z, x*z, x, z, 1.0])
            vals.append(psi[ii, jj])

    A_ls = np.array(rows, dtype=np.float64)
    b_ls = np.array(vals, dtype=np.float64)

    coeff, *_ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
    a, bq, c, d, e, g = coeff

    # Stationary point of quadratic surface
    # df/dx = 2a x + c z + d = 0
    # df/dz = c x + 2b z + e = 0
    M = np.array([[2*a, c],
                  [c,  2*bq]], dtype=np.float64)
    rhs = -np.array([d, e], dtype=np.float64)

    # If Hessian is nearly singular -> fallback
    if np.linalg.cond(M) > 1e10:
        return R[i0], Z[j0], psi[i0, j0]

    x_star, z_star = np.linalg.solve(M, rhs)

    psi_axis = a*x_star*x_star + bq*z_star*z_star + c*x_star*z_star + d*x_star + e*z_star + g
    return x0 + x_star, z0 + z_star, psi_axis


def compute_psibar(psi, psi_axis, psi_b, inside_mask):
    """
    Compute normalized poloidal flux ψ̄ inside the plasma:

        ψ̄ = (ψ - ψ_axis) / (ψ_b - ψ_axis)

    Convention used:
      • ψ = ψ_axis at axis  -> ψ̄ = 0
      • ψ = ψ_b at boundary -> ψ̄ = 1

    Outside the mask we return ψ̄ = 1 (not physically meaningful, just convenient).

    Returns
    -------
    psibar : array same shape as psi
    denom : (ψ_b - ψ_axis) used for normalization
    """
    denom = (psi_b - psi_axis)
    if abs(denom) < 1e-14:
        denom = np.sign(denom) * 1e-14 if denom != 0 else 1e-14

    psibar = np.ones_like(psi, dtype=np.float64)
    psibar[inside_mask] = (psi[inside_mask] - psi_axis) / denom
    psibar[inside_mask] = np.clip(psibar[inside_mask], 0.0, 1.0)
    return psibar, denom


def profiles_from_psibar(psibar, denom, inside_mask, p0=2e4, alpha=2.0, F0=1.0, delta=0.05, beta=2.0):
    """
    Compute p'(ψ) and F(ψ)F'(ψ) from simple analytic profiles defined in ψ̄.

    We define:
      p(ψ̄) = p0 (1 - ψ̄)^alpha
      F(ψ̄) = F0 (1 - delta * ψ̄^beta)

    Then use chain rule:
      dp/dψ = (dp/dψ̄) * (dψ̄/dψ),  where dψ̄/dψ = 1/denom

    Parameters
    ----------
    psibar : array
        Normalized flux ψ̄.
    denom : float
        ψ_b - ψ_axis (normalization denominator).
    inside_mask : bool array
        Only inside points get non-zero sources.

    Returns
    -------
    pprime : array
        p'(ψ) on the grid (zero outside plasma).
    FFprime : array
        F(ψ)F'(ψ) on the grid (zero outside plasma).
    """
    dpsibar_dpsi = 1.0 / denom

    pprime = np.zeros_like(psibar, dtype=np.float64)
    FFprime = np.zeros_like(psibar, dtype=np.float64)

    pb = psibar[inside_mask]

    # Pressure derivative: dp/dψ̄
    dp_dpsibar = -p0 * alpha * (1.0 - pb) ** (alpha - 1.0)
    pprime[inside_mask] = dp_dpsibar * dpsibar_dpsi

    # Toroidal field function derivative: F*F'
    F = F0 * (1.0 - delta * pb**beta)
    dF_dpsibar = F0 * (-delta * beta * pb ** (beta - 1.0))
    Fprime = dF_dpsibar * dpsibar_dpsi
    FFprime[inside_mask] = F * Fprime

    return pprime, FFprime


# =============================================================================
# GS solver (fixed boundary, masked Dirichlet, Picard iteration)
# =============================================================================
def solve_gs_fixed_shaped_boundary(
    Rmin=1.0, Rmax=3.0, Zmin=-1.5, Zmax=1.5,
    NR=257, NZ=257,
    psi_b=0.0,
    max_iter=80,
    omega=0.5,
    tol=1e-7,
    # LCFS (Miller)
    R0=1.7, a=0.5, kappa=1.7, delta=0.33,
    # profile parameters (toy)
    p0=2e4, alpha=2.0, F0=1.0, deltaF=0.05, beta=2.0
):
    """
    Solve Grad–Shafranov on a rectangular grid with an embedded LCFS boundary.

    Returns
    -------
    R, Z : 1D arrays
    psi : 2D array ψ(R,Z)
    (Rb, Zb) : boundary polygon
    inside_mask : bool mask for plasma region
    """
    # --- Build grid ---
    R = np.linspace(Rmin, Rmax, NR)
    Z = np.linspace(Zmin, Zmax, NZ)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")

    # --- Choose boundary polygon and build mask ---
    # If you want diverted boundary (geometric), you could switch this:
    # Rb, Zb = diverted_lcfs_polygon(R0=R0, a=a, kappa=kappa, delta=delta)
    Rb, Zb = miller_lcfs(R0=R0, a=a, kappa=kappa, delta=delta, ntheta=2000)

    inside_mask = build_inside_mask(R, Z, Rb, Zb)

    # --- Build linear operator (constant during Picard iterations) ---
    A, dR, dZ = build_delta_star_matrix_masked(R, Z, inside_mask)

    # --- Initial guess ---
    # Start with ψ=ψ_b everywhere; seed a small dip at (R0,0) to create an "axis"
    psi = np.full((NR, NZ), psi_b, dtype=np.float64)

    i0 = int(np.argmin(np.abs(R - R0)))
    j0 = int(np.argmin(np.abs(Z - 0.0)))
    if inside_mask[i0, j0]:
        psi[i0, j0] = psi_b - 1.0

    # Outside LCFS is clamped to ψ_b by Dirichlet rows
    psi[~inside_mask] = psi_b

    # --- Picard iteration loop ---
    for it in range(max_iter):
        # Axis and normalized flux based on current iterate
        Rax, Zax, psi_axis = find_magnetic_axis_subgrid(R, Z, psi, inside_mask)
        psibar, denom = compute_psibar(psi, psi_axis, psi_b, inside_mask)

        # Compute RHS source terms (flux functions)
        pprime, FFprime = profiles_from_psibar(
            psibar, denom, inside_mask, p0=p0, alpha=alpha, F0=F0, delta=deltaF, beta=beta
        )

        # GS equation:
        #   Δ*ψ = -μ0 R^2 p'(ψ) - F F'(ψ)
        rhs = (-MU0 * (RR**2) * pprime - FFprime)

        # Enforce Dirichlet RHS on all constrained nodes
        rhs[~inside_mask] = psi_b
        rhs[0, :] = psi_b; rhs[-1, :] = psi_b
        rhs[:, 0] = psi_b; rhs[:, -1] = psi_b

        b = rhs.reshape(-1)

        # Solve A ψ_new = b with BiCGSTAB
        psi_new_vec, info = bicgstab(A, b, x0=psi.reshape(-1), tol=1e-10, maxiter=800)
        psi_new = psi_new_vec.reshape(NR, NZ)

        # Under-relaxation improves stability of Picard iteration
        psi_next = (1.0 - omega) * psi + omega * psi_new
        psi_next[~inside_mask] = psi_b  # numerical clamp (identity rows already enforce it)

        # Convergence measured only inside plasma
        diff = (
            np.linalg.norm((psi_next[inside_mask] - psi[inside_mask]).ravel())
            / (np.linalg.norm(psi[inside_mask].ravel()) + 1e-30)
        )

        psi = psi_next
        if diff < tol:
            break

    return R, Z, psi, (Rb, Zb), inside_mask


# =============================================================================
# Run folder utilities + HDF5 saving
# =============================================================================
def _next_run_dir(base_dir: str, prefix: str = "sim") -> Path:
    """
    Create a new run subdirectory:

        base_dir / f"{prefix}{XXX}"

    where XXX is an incrementing integer with leading zeros starting at 001.
    Example:
        runs/sim001, runs/sim002, ...

    Returns
    -------
    run_dir : pathlib.Path
        The newly created directory.
    """
    base = Path(base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)

    # Scan existing run directories to determine the next index
    nums = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            suf = p.name[len(prefix):]
            if suf.isdigit():
                nums.append(int(suf))

    n = (max(nums) + 1) if nums else 1
    run_dir = base / f"{prefix}{n:03d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def save_gs_h5(
    out_dir: str,
    R: np.ndarray,
    Z: np.ndarray,
    psi: np.ndarray,
    inside_mask: np.ndarray,
    Rb: np.ndarray = None,
    Zb: np.ndarray = None,
    metadata: dict = None,
    prefix: str = "sim",
    filename: str = "gs_solution.h5",
) -> Path:
    """
    Save GS results to an HDF5 file in a new run folder.

    Output layout:
        out_dir / simXXX / gs_solution.h5

    Stored datasets:
        R, Z, psi, inside_mask (as uint8 for portability)
        Rb, Zb (if provided)

    Stored attributes:
        anything in `metadata` (scalars/strings)

    Returns
    -------
    out_path : pathlib.Path
        Full path to the saved HDF5 file.
    """
    run_dir = _next_run_dir(out_dir, prefix=prefix)
    out_path = run_dir / filename

    # Store mask as uint8 for compactness and compatibility
    inside_u8 = inside_mask.astype(np.uint8)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("R", data=R)
        f.create_dataset("Z", data=Z)
        f.create_dataset("psi", data=psi)
        f.create_dataset("inside_mask", data=inside_u8)

        if Rb is not None and Zb is not None:
            f.create_dataset("Rb", data=Rb)
            f.create_dataset("Zb", data=Zb)

        if metadata:
            for k, v in metadata.items():
                # HDF5 attrs support simple types; fallback to string otherwise
                try:
                    f.attrs[k] = v
                except TypeError:
                    f.attrs[k] = str(v)

    print(f"Saved GS solution to: {out_path}")
    return out_path


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    # Base directory where simXXX folders are created
    BASE_OUTPUT_DIR = "/Users/leon/Desktop/python_skripte/tokamak_design/old/runs"  # <-- change as desired

    # Solve equilibrium
    R, Z, psi, (Rb, Zb), inside_mask = solve_gs_fixed_shaped_boundary()

    # Metadata saved as HDF5 attributes (useful for later bookkeeping)
    meta = {
        "psi_b": 0.0,
        # You can add parameters you used (recommended):
        # "Rmin": Rmin, "Rmax": Rmax, "Zmin": Zmin, "Zmax": Zmax, "NR": NR, "NZ": NZ,
        # "R0": R0, "a": a, "kappa": kappa, "delta": delta,
        # "p0": p0, "alpha": alpha, "F0": F0, "deltaF": deltaF, "beta": beta,
    }

    # Save to runs/simXXX/gs_solution.h5
    save_gs_h5(
        out_dir=BASE_OUTPUT_DIR,
        R=R,
        Z=Z,
        psi=psi,
        inside_mask=inside_mask,
        Rb=Rb,
        Zb=Zb,
        metadata=meta,
        prefix="sim",
        filename="gs_solution.h5",
    )

    # Optional quick plot (for sanity)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    plt.figure()
    plt.plot(Rb, Zb, linewidth=2)
    cs = plt.contour(RR, ZZ, psi, levels=20)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("R [m]")
    plt.ylabel("Z [m]")
    plt.title("Fixed-boundary Grad–Shafranov equilibrium (ψ contours)")
    plt.show()
