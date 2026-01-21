"""
constants.py
=============

Physical constants and global numerical parameters used throughout
the tokamak design workflow.

Philosophy:
------------
• Include more constants than strictly necessary
• Avoid magic numbers elsewhere in the code
• Keep everything in SI units
• Document sources and meanings clearly

Primary sources:
----------------
• CODATA 2018
• Plasma physics standard conventions
"""

import numpy as np

# ============================================================
# MATHEMATICAL CONSTANTS
# ============================================================

PI = np.pi
TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
HALF_PI = 0.5 * np.pi

E = np.e

# ============================================================
# FUNDAMENTAL PHYSICAL CONSTANTS (SI UNITS)
# ============================================================

# Speed of light in vacuum [m/s]
C_LIGHT = 2.99792458e8

# Elementary charge [C]
E_CHARGE = 1.602176634e-19

# Boltzmann constant [J/K]
K_BOLTZMANN = 1.380649e-23

# Planck constant [J·s]
H_PLANCK = 6.62607015e-34

# Reduced Planck constant [J·s]
HBAR = H_PLANCK / TWO_PI

# Electron mass [kg]
M_ELECTRON = 9.1093837015e-31

# Proton mass [kg]
M_PROTON = 1.67262192369e-27

# Neutron mass [kg]
M_NEUTRON = 1.67492749804e-27

# Atomic mass unit [kg]
AMU = 1.66053906660e-27

# ============================================================
# ELECTROMAGNETIC CONSTANTS
# ============================================================

# Vacuum permeability μ0 [H/m = N/A^2]
MU0 = 4.0e-7 * np.pi

# Vacuum permittivity ε0 [F/m]
EPS0 = 1.0 / (MU0 * C_LIGHT**2)

# Coulomb constant k = 1 / (4π ε0)
K_COULOMB = 1.0 / (4.0 * np.pi * EPS0)

# ============================================================
# PLASMA PHYSICS CONSTANTS
# ============================================================

# Electron charge (signed)
QE = -E_CHARGE

# Ion charge (proton)
QI = E_CHARGE

# Gyromagnetic ratio (electron) [rad/s/T]
GAMMA_E = E_CHARGE / (2.0 * M_ELECTRON)

# Classical electron radius [m]
R_ELECTRON = E_CHARGE**2 / (4.0 * np.pi * EPS0 * M_ELECTRON * C_LIGHT**2)

# Alfvén speed coefficient:
# v_A = B / sqrt(MU0 * rho)
# -> keep MU0 explicit elsewhere

# ============================================================
# FUSION-SPECIFIC CONSTANTS
# ============================================================

# Deuterium mass [kg]
M_DEUTERIUM = 2.013553212745 * AMU

# Tritium mass [kg]
M_TRITIUM = 3.01550071621 * AMU

# Alpha particle mass [kg]
M_ALPHA = 4.001506179127 * AMU

# D-T fusion energy [J]
# 17.6 MeV -> Joules
DT_FUSION_ENERGY = 17.6e6 * E_CHARGE

# Alpha particle birth energy [J] (3.5 MeV)
ALPHA_BIRTH_ENERGY = 3.5e6 * E_CHARGE

# ============================================================
# TOKAMAK / MHD CONVENTIONS
# ============================================================

# Grad-Shafranov operator sign convention:
# Δ*ψ = - MU0 * R * j_phi
GS_SIGN = -1.0

# Preferred psi convention:
# psi_lcfs = 0
# psi more negative inside plasma
DEFAULT_PSI_LCFS = 0.0

# Numerical zero tolerance
EPS = 1e-14

# ============================================================
# UNIT CONVERSION HELPERS
# ============================================================

EV_TO_J = E_CHARGE
J_TO_EV = 1.0 / E_CHARGE

KEV_TO_J = 1.0e3 * E_CHARGE
MEV_TO_J = 1.0e6 * E_CHARGE

TESLA_TO_GAUSS = 1.0e4
GAUSS_TO_TESLA = 1.0e-4

# ============================================================
# DEFAULT NUMERICAL PARAMETERS (SAFE STARTING VALUES)
# ============================================================

# Finite difference
DEFAULT_DERIV_ORDER = 2

# Iterative solver
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1.0e-8
DEFAULT_RELAX = 0.5

# Contour extraction
DEFAULT_CONTOUR_POINTS = 360

# ============================================================
# SANITY CHECK
# ============================================================

def _self_test():
    """
    Basic internal consistency checks.
    Called manually if needed.
    """
    # Check c^2 = 1/(mu0*eps0)
    c_test = 1.0 / np.sqrt(MU0 * EPS0)
    assert np.isclose(c_test, C_LIGHT, rtol=1e-10)

    # Check elementary charge sign
    assert QE < 0.0
    assert QI > 0.0

    print("constants.py self-test passed")


if __name__ == "__main__":
    _self_test()
