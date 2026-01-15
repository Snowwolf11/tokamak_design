"""
units.py
========

Central place for physical units and future unit handling.

Current status
--------------
• Minimal implementation (no external dependency)
• Assumes SI units everywhere
• Acts as documentation + extension point

Future
------
This module is designed so that you can later integrate `pint`
or another unit library without changing the rest of the codebase.

Design rule
-----------
ALL unit logic goes here.
No script should hard-code conversions elsewhere.
"""

# ============================================================
# BASE UNITS (DOCUMENTATION)
# ============================================================

# Length
m = 1.0          # meter
cm = 1e-2
mm = 1e-3
km = 1e3

# Time
s = 1.0          # second
ms = 1e-3
us = 1e-6

# Current
A = 1.0          # Ampere
kA = 1e3
MA = 1e6

# Magnetic field
T = 1.0          # Tesla
mT = 1e-3

# Pressure
Pa = 1.0
kPa = 1e3
MPa = 1e6

# Energy
J = 1.0
kJ = 1e3
MJ = 1e6

# ============================================================
# PLASMA-SPECIFIC CONVENIENCE
# ============================================================

# Physical constants (you already have them in constants.py,
# but they’re repeated here for semantic clarity)
mu0 = 4e-7 * 3.141592653589793  # vacuum permeability

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_SI(value, unit_factor):
    """
    Convert a value to SI units.

    Example
    -------
    >>> to_SI(10, cm)
    0.1

    This is trivial now, but later this function will:
    • accept pint quantities
    • enforce dimensionality
    """
    return value * unit_factor


def from_SI(value, unit_factor):
    """
    Convert a value from SI to desired unit.

    Example
    -------
    >>> from_SI(0.1, cm)
    10
    """
    return value / unit_factor


# ============================================================
# FUTURE PLACEHOLDERS
# ============================================================

def enable_pint():
    """
    Placeholder hook.

    In the future:
    • import pint
    • create UnitRegistry
    • expose ureg globally
    """
    raise NotImplementedError(
        "Pint integration not implemented yet."
    )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("Testing units.py")

    assert to_SI(10, cm) == 0.1
    assert from_SI(0.1, cm) == 10

    print("units.py self-test passed")
