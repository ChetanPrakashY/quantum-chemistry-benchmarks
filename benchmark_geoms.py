"""Molecules, geometries and active spaces used by the benchmark."""
import numpy as np


def geom(name, s):
    """Geometries scaled by s along the bonds (s = 1: near equilibrium)."""
    if name == "H2":
        return f"H 0 0 0; H 0 0 {0.74 * s}"
    if name == "LiH":
        return f"Li 0 0 0; H 0 0 {1.60 * s}"
    if name == "BeH2":
        return f"H 0 0 {-1.33 * s}; Be 0 0 0; H 0 0 {1.33 * s}"
    if name == "H2O":
        r, a = 0.96 * s, np.radians(104.5 / 2)
        return f"O 0 0 0; H {r*np.sin(a)} 0 {r*np.cos(a)}; H {-r*np.sin(a)} 0 {r*np.cos(a)}"


ACTIVE = {"H2": (2, 2), "LiH": (3, 2), "BeH2": (4, 4), "H2O": (4, 4)}   # (orbitals, electrons)
