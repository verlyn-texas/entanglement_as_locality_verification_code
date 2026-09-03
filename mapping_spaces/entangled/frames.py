"""qD reference frames for an entangled electron pair, built on the shear
model of ``mapping_spaces/transformation.py``.

Dictionary between the two solutions
------------------------------------
* qD  == the third coordinate z.
* lab (measuring apparatus) == frame 3, at z = 0.
* particle A == object 1 at z = -1; particle B == object 2 at z = +1.
* y == lab time (Galilean, absolute), x == lab position along the axis
  on which the pair separates, a == 1/(half the separation speed): the lab
  sees A at x = -y/a and B at x = +y/a, both created at x = 0 when y = 0.
* In A's frame (1) and B's frame (2) both particles sit at x = 0 for all y:
  they are co-located ("local to each other").  The lab is seen moving.

Continuous qD
-------------
Particles placed at z = -c and z = +c (0 <= c <= 1) are no longer exactly
co-located in each other's frames.  Two conventions (row R05):
* pair at rest in its own frames (``separation_in_frame`` with zA=-c, zB=c):
  with the quadratic potentials the residual rate is (1 - c^2) of the lab's,
  so the "locality fraction" is c^2 (``locality_fraction``); this depends on
  the interpolant and makes the lab-frame separation speed c-dependent;
* lab tracks pinned to x = -/+ y/a (general paths, transformation.md §4):
  the interpolant cancels and the fraction is c itself.
Row R05 ties c to the pair's concurrence and adopts the pinned convention.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces import transformation as T

LAB, A, B = 3, 1, 2
Z = {A: -1.0, B: 1.0, LAB: 0.0}


def beta(k: int, l: int, a: float, z) -> np.ndarray:
    """Shear coefficient from frame k to frame l at height z (vectorised)."""
    z = np.asarray(z, dtype=float)
    return T.shear_potential(l, a)(z) - T.shear_potential(k, a)(z)  # type: ignore[operator]


def to_frame(k: int, l: int, a: float, x, y, z):
    """Map points (x, y, z) from frame k to frame l."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    return x + beta(k, l, a, z) * y, y, z


def lab_tracks(a: float, y):
    """Ideal lab tracks x_A(y), x_B(y) of a pair created at the origin."""
    y = np.asarray(y, dtype=float)
    return -y / a, y / a


def path_in_frame(obj: int, k: int, a: float, p, y):
    """A particle following x = p(y) in its own frame, seen in frame k."""
    y = np.asarray(y, dtype=float)
    return p(y) + beta(obj, k, a, Z[obj]) * y


def separation_in_frame(k: int, a: float, y, zA: float = -1.0, zB: float = 1.0,
                        pA=None, pB=None):
    """x_B - x_A as seen in frame k, for particles at heights zA, zB that
    follow paths pA, pB (default: at rest) in their own frames."""
    y = np.asarray(y, dtype=float)
    pA = pA or (lambda t: 0 * t)
    pB = pB or (lambda t: 0 * t)
    xA = pA(y) + beta(A, k, a, zA) * y
    xB = pB(y) + beta(B, k, a, zB) * y
    return xB - xA


def locality_fraction(c: float) -> float:
    """Fraction of the lab-frame separation rate removed in the particles'
    frames when they sit at z = -c, +c: 1 - beta_12(c)/beta_12(1)... in
    closed form c^2 (see module docstring)."""
    return float(c * c)


def lab_time_order(yA: float, yB: float) -> int:
    """Which measurement is first in *every* frame: y is untouched by all
    frame maps, so the ordering is absolute.  Returns A (1) or B (2), or 0 if
    simultaneous."""
    if yA < yB:
        return A
    if yB < yA:
        return B
    return 0
