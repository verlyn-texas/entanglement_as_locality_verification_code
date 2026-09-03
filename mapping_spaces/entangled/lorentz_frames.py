"""Pass 2 — a Lorentz-covariant version of the qD frames (1+1 dimensions, c = 1).

Instead of a Galilean shear, the map from the lab to frame k at height z is
a Lorentz boost with rapidity psi_k(z):

    (t', x') = Lambda(psi_k(z)) (t, x),   Lambda(psi) = [[cosh, -sinh], [-sinh, cosh]],

so the map from frame k to frame l at height z is Lambda(psi_l(z) - psi_k(z))
and composition is automatic (rapidities add).  Pinning mirrors
graph_frames.py: particle j (lab velocity v_j) is seen in frame k moving at

    u_kj = (1 - w_kj) * (v_j - v_k) / (1 - v_j v_k)          (relativistic
                                                             relative velocity,
                                                             scaled by locality)

which fixes psi_k(z_j) = artanh(v_j) - artanh(u_kj); psi_k(z_k) = artanh(v_k);
psi_k(0) = artanh(v_k) (the lab is seen moving at -v_k); psi_L = 0.  Between
pinned heights the rapidity is interpolated by the Lagrange polynomial.

Consequences (row R14):
* a partner j (w = 1) is at rest at x' = 0 in frame k, and the frame time of
  an event on j's worldline is j's PROPER time since the creation event;
* the ordering of the two measurement events in the pair's frame is by
  proper time, which is invariant under a global boost of the lab;
* for |v| << 1 the boost reduces to the shear of transformation.md with
  sigma_k(z) = sigma_L(z) - tanh psi_k(z) (checked numerically).
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled.graph_frames import LAB, LocalityGraph, lagrange


def rel_velocity(vj: float, vk: float) -> float:
    return (vj - vk) / (1.0 - vj * vk)


def boost(psi: float) -> np.ndarray:
    c, s = np.cosh(psi), np.sinh(psi)
    return np.array([[c, -s], [-s, c]])


def rapidity_potential(g: LocalityGraph, k):
    """psi_k(z) as a callable."""
    zs, vals = [], []
    heights = sorted({0.0, *(p.z for p in g.particles.values())})
    for z in heights:
        if k == LAB:
            vals.append(0.0)
        elif z == 0.0:
            vals.append(np.arctanh(g.particles[k].v))
        else:
            j = next(n for n, p in g.particles.items() if p.z == z)
            if j == k:
                vals.append(np.arctanh(g.particles[k].v))
            else:
                u = (1.0 - g.weight(k, j)) * rel_velocity(g.particles[j].v, g.particles[k].v)
                vals.append(np.arctanh(g.particles[j].v) - np.arctanh(u))
        zs.append(z)
    return lagrange(zs, vals)


def to_frame(g: LocalityGraph, k, z: float, t, x):
    """Lab event (t, x) at height z expressed in frame k."""
    L = boost(float(rapidity_potential(g, k)(z)))
    tx = L @ np.array([t, x], dtype=float)
    return float(tx[0]), float(tx[1])


def frame_to_frame(g: LocalityGraph, k, l, z: float, t, x):
    L = boost(float(rapidity_potential(g, l)(z) - rapidity_potential(g, k)(z)))
    tx = L @ np.array([t, x], dtype=float)
    return float(tx[0]), float(tx[1])


def velocity_in_frame(g: LocalityGraph, j, k) -> float:
    """Velocity of particle j in frame k (from the boost of its worldline)."""
    pj = g.particles[j]
    t1, x1 = to_frame(g, k, pj.z, 1.0, pj.v)
    t0, x0 = to_frame(g, k, pj.z, 0.0, 0.0)
    return (x1 - x0) / (t1 - t0)


def proper_time(v: float, t: float) -> float:
    return t * np.sqrt(1.0 - v * v)


def measurement_events_in_frame(g: LocalityGraph, k, i, j, ti, tj):
    """The two measurement events (particle i at lab time ti on its worldline,
    particle j at tj) expressed in frame k: returns ((t'_i, x'_i), (t'_j, x'_j))."""
    pi, pj = g.particles[i], g.particles[j]
    return (to_frame(g, k, pi.z, ti, pi.v * ti), to_frame(g, k, pj.z, tj, pj.v * tj))


def order_in_pair_frame(g: LocalityGraph, i, j, ti, tj):
    """Which measurement is first in the pair's own frame: compares the frame
    times, which for local partners are the proper times."""
    (ti_, _), (tj_, _) = measurement_events_in_frame(g, i, i, j, ti, tj)
    if abs(ti_ - tj_) < 1e-15:
        return None
    return i if ti_ < tj_ else j


def boosted_lab(g: LocalityGraph, u: float) -> LocalityGraph:
    """The same physical situation described by a lab moving at velocity u
    (all lab velocities composed relativistically)."""
    h = LocalityGraph({}, dict(g.w))
    for p in g.particles.values():
        h.particles[p.name] = type(p)(p.name, p.z, rel_velocity(p.v, u), p.x0)
    return h
