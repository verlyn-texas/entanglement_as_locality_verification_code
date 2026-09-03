"""R14 — Lorentz-covariant qD frames and proper-time ordering.

THIS ROW IS RELATIVISTIC (1+1 dimensions, c = 1); every other row of the
project is non-relativistic.  It uses the shared module
``lorentz_frames.py`` (mechanism M11) unchanged and adds only the helpers
that the numbered claims of rows/R14_lorentz_frames.md need.

Claims (numbered as in the row document):
 1. Frame k at height z is the boost by psi_k(z); k -> l is the boost by
    psi_l(z) - psi_k(z), so composition holds at every height.  For a pair
    with w = 1 each partner is at rest at x' = 0 in the other's frame, for
    asymmetric lab velocities (v_A = -0.3, v_B = +0.8) as well as symmetric.
 2. The frame-A time of an event at lab time t on partner j's worldline is
    tau_j = t sqrt(1 - v_j^2): the pair frame orders the two measurements by
    proper time since creation.  The ordering (and the tau's themselves) are
    invariant under a global boost of the lab.
 3. Proper-time and lab-time orderings can disagree: for t_A < t_B they do
    iff t_B/t_A < gamma_B/gamma_A (the later-in-lab particle is the faster
    one).  For (t_A, t_B) uniform on the unit square the disagreeing
    fraction is (r - 1)/(2r), r = gamma_max/gamma_min; zero for symmetric
    velocities.  Disagreement occurs only for spacelike-separated events —
    when the events are causally connected the proper-time order equals the
    causal order (reverse triangle inequality).
 4. The pair frame is not a global Lorentz frame: heights z_A and z_B are
    boosted by different rapidities.  An unrelated particle C (w = 0) moves in
    frame A at the relativistic relative velocity, and an event on its
    worldline gets frame-A time gamma_A (1 - v_A v_C) t — A's ordinary
    rest-frame time, not C's proper time.
 5. Non-relativistic limit: sigma_k(z) := sigma_L(z) - tanh psi_k(z) tends to
    the shear potentials of transformation.md with relative error ∝ v^2.
 6. Spin statistics are order-independent: sequential projection in either
    order reproduces the QM box, S = 2 sqrt 2, no signalling.
 7. Multisimultaneity (ordering by each device's rest frame, refuted by
    L-B2) versus M11 (ordering by the particles' proper times): the former
    flips with device velocity inside a window |Delta t| < beta Delta x; the
    latter does not depend on any device velocity.
"""
from __future__ import annotations

import itertools

import numpy as np

from mapping_spaces import transformation as T
from mapping_spaces.entangled import graph_frames as G, lorentz_frames as LZ, qm

C_SI = 299_792_458.0  # m/s, only to convert ledger device speeds to beta


def gamma(v: float) -> float:
    return 1.0 / np.sqrt(1.0 - v * v)


# ------------------------------------------------------------ configurations
def pair(vA: float, vB: float, zA: float = -1.0, zB: float = 1.0, w: float = 1.0) -> G.LocalityGraph:
    """Two particles created at the lab origin, partners with weight w."""
    g = G.LocalityGraph()
    g.add(G.Particle("A", zA, vA)).add(G.Particle("B", zB, vB))
    g.link("A", "B", w)
    return g


def with_third(g: G.LocalityGraph, name: str, z: float, v: float, w: float = 0.0,
               partner: str = "A") -> G.LocalityGraph:
    """Add a third particle at height z; w = 0 leaves it unrelated to ``partner``."""
    h = G.LocalityGraph(dict(g.particles), dict(g.w))
    h.add(G.Particle(name, z, v))
    if w > 0:
        h.link(partner, name, w)
    return h


# ------------------------------------------------------------------ claim 1
def composition_defect(g, k, l, m, z: float, t: float, x: float) -> float:
    """| (k->m) - (l->m) o (k->l) | at height z for the event (t, x)."""
    via = LZ.frame_to_frame(g, l, m, z, *LZ.frame_to_frame(g, k, l, z, t, x))
    direct = LZ.frame_to_frame(g, k, m, z, t, x)
    return float(np.hypot(via[0] - direct[0], via[1] - direct[1]))


def lab_velocity_in_frame(g, k) -> float:
    """Velocity of the lab origin (worldline x = 0 at height 0) in frame k."""
    t1, x1 = LZ.to_frame(g, k, 0.0, 1.0, 0.0)
    t0, x0 = LZ.to_frame(g, k, 0.0, 0.0, 0.0)
    return (x1 - x0) / (t1 - t0)


def partner_rest(g, i: str, j: str, ti: float, tj: float) -> dict:
    """Velocities and event positions of the pair in each other's frames."""
    (_, xi_in_i), (_, xj_in_i) = LZ.measurement_events_in_frame(g, i, i, j, ti, tj)
    (_, xi_in_j), (_, xj_in_j) = LZ.measurement_events_in_frame(g, j, i, j, ti, tj)
    return {
        "v_j_in_i": LZ.velocity_in_frame(g, j, i),
        "v_i_in_j": LZ.velocity_in_frame(g, i, j),
        "x_in_i": (xi_in_i, xj_in_i),
        "x_in_j": (xi_in_j, xj_in_j),
    }


# ------------------------------------------------------------------ claim 2
def frame_times(g, k, i: str, j: str, ti: float, tj: float) -> tuple[float, float]:
    (ti_, _), (tj_, _) = LZ.measurement_events_in_frame(g, k, i, j, ti, tj)
    return ti_, tj_


def boosted_lab_times(g, u: float, i: str, j: str, ti: float, tj: float) -> tuple[float, float]:
    """Lab times of the same two measurement events as read by a lab moving
    at u (the recipe of tests/test_entangled_graph_frames.py::Lorentz)."""
    gam = gamma(u)
    vi, vj = g.particles[i].v, g.particles[j].v
    return gam * ti * (1.0 - u * vi), gam * tj * (1.0 - u * vj)


def ordering_under_boost(g, u: float, i: str, j: str, ti: float, tj: float):
    """(first in the pair frame, proper times) described from a lab moving at u."""
    h = LZ.boosted_lab(g, u)
    ti_u, tj_u = boosted_lab_times(g, u, i, j, ti, tj)
    return LZ.order_in_pair_frame(h, i, j, ti_u, tj_u), frame_times(h, i, i, j, ti_u, tj_u)


# ------------------------------------------------------------------ claim 3
def lab_order(ti: float, tj: float, i: str = "A", j: str = "B"):
    if ti == tj:
        return None
    return i if ti < tj else j


def proper_order(ti: float, tj: float, vi: float, vj: float, i: str = "A", j: str = "B"):
    taui, tauj = LZ.proper_time(vi, ti), LZ.proper_time(vj, tj)
    if abs(taui - tauj) < 1e-15:
        return None
    return i if taui < tauj else j


def orderings_disagree(tA, tB, vA, vB) -> np.ndarray:
    """Vectorised: lab order != proper-time order (strict, both defined)."""
    tA, tB = np.asarray(tA, float), np.asarray(tB, float)
    lab = np.sign(tB - tA)
    prop = np.sign(tB * np.sqrt(1 - vB * vB) - tA * np.sqrt(1 - vA * vA))
    return (lab != prop) & (lab != 0) & (prop != 0)


def disagreement_ratio_bound(vA: float, vB: float) -> float:
    """r = gamma_max/gamma_min: for t_A < t_B the orderings disagree iff
    1 < t_B/t_A < gamma_B/gamma_A (and symmetrically), i.e. the ratio of lab
    times lies below r."""
    gA, gB = gamma(vA), gamma(vB)
    return float(max(gA, gB) / min(gA, gB))


def disagreement_fraction_closed_form(vA: float, vB: float) -> float:
    """Area of the wedge {1 < t_B/t_A < r} ∪ {1 < t_A/t_B < r} inside the unit
    square (only one of the two wedges is non-empty): (r - 1)/(2 r)."""
    r = disagreement_ratio_bound(vA, vB)
    return float((r - 1.0) / (2.0 * r))


def disagreement_fraction_mc(vA: float, vB: float, n: int = 20000, seed: int = 14) -> float:
    rng = np.random.default_rng(seed)
    tA, tB = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    return float(orderings_disagree(tA, tB, vA, vB).mean())


def is_spacelike(tA, tB, vA, vB) -> np.ndarray:
    """The two measurement events are spacelike separated."""
    tA, tB = np.asarray(tA, float), np.asarray(tB, float)
    dt, dx = tB - tA, vB * tB - vA * tA
    return np.abs(dx) > np.abs(dt)


def causal_order_respected(n: int = 20000, seed: int = 141) -> tuple[bool, int, int]:
    """Random events with random velocities: whenever the events are causally
    (timelike or lightlike) connected, the proper-time order equals the lab
    (= causal) order.  Returns (all agree, #causal cases, #disagreements)."""
    rng = np.random.default_rng(seed)
    vA, vB = rng.uniform(-0.95, 0.95, n), rng.uniform(-0.95, 0.95, n)
    tA, tB = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    causal = ~is_spacelike(tA, tB, vA, vB)
    lab = np.sign(tB - tA)
    prop = np.sign(tB * np.sqrt(1 - vB**2) - tA * np.sqrt(1 - vA**2))
    bad = causal & (lab != prop)
    return bool(not bad.any()), int(causal.sum()), int(bad.sum())


# ------------------------------------------------------------------ claim 4
def rapidity_at(g, k, z: float) -> float:
    return float(LZ.rapidity_potential(g, k)(z))


def third_particle_frame_time(g, k: str, c: str, t: float) -> float:
    """Frame-k time of the event at lab time t on particle c's worldline."""
    pc = g.particles[c]
    t_, _ = LZ.to_frame(g, k, pc.z, t, pc.v * t)
    return t_


def lorentz_rest_frame_time(vk: float, vc: float, t: float) -> float:
    """Time of the event (t, v_c t) in the ordinary rest frame of k."""
    return float(gamma(vk) * (1.0 - vk * vc) * t)


# ------------------------------------------------------------------ claim 5
def nonrelativistic_error(eps: float, z=None) -> float:
    """max_z |sigma_L - tanh psi_A - sigma_1| / max_z |sigma_1| for the pass-1
    pair with v = -/+eps (a = 1/eps)."""
    z = np.linspace(-1, 1, 41) if z is None else np.asarray(z, float)
    g = G.pass1_pair(a=1 / eps)
    sigma_A = g.potential(G.LAB)(z) - np.tanh(LZ.rapidity_potential(g, "A")(z))
    ref = T.shear_potential(1, 1 / eps)(z)
    return float(np.max(np.abs(sigma_A - ref)) / np.max(np.abs(ref)))


def nonrelativistic_sweep(epsilons=(1e-1, 1e-2, 1e-3)) -> dict:
    return {eps: nonrelativistic_error(eps) for eps in epsilons}


# ------------------------------------------------------------------ claim 6
def sequential_box(state, settings_A, settings_B, first: str = "A") -> np.ndarray:
    """Box from projecting the ``first`` wing, then measuring the other on
    the projected state (the model's rule, with the order given by the
    pair frame's proper times)."""
    box = np.zeros((2, 2, 2, 2))
    for x, y, ia, ib in itertools.product(range(2), repeat=4):
        if first == "A":
            rho1, p1 = qm.project(state, settings_A[x], qm.OUTCOME[ia], "A")
            p2 = qm.marginal_probability(rho1, settings_B[y], qm.OUTCOME[ib], "B")
        else:
            rho1, p1 = qm.project(state, settings_B[y], qm.OUTCOME[ib], "B")
            p2 = qm.marginal_probability(rho1, settings_A[x], qm.OUTCOME[ia], "A")
        box[x, y, ia, ib] = p1 * p2
    return box


# ------------------------------------------------------------------ claim 7
def device_frame_first(tA: float, xA: float, tB: float, xB: float, uA: float, uB: float):
    """Multisimultaneity: which event is first in the rest frame of device A
    (moving at uA) and in that of device B (moving at uB).  Returns the two
    answers; ('A', 'B') is the 'before-before' configuration."""
    def first_in(u):
        gam = gamma(u)
        tA_, tB_ = gam * (tA - u * xA), gam * (tB - u * xB)
        return lab_order(tA_, tB_)
    return first_in(uA), first_in(uB)


def before_before_window(beta: float, dx: float) -> float:
    """|Delta t| below which two events dx apart (c = 1) have opposite order
    in frames moving at -/+beta along the axis: beta * dx."""
    return float(beta * dx)
