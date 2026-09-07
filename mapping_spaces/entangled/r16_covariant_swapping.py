"""R16 — Lorentz-covariant graph dynamics: ordering for a swapped pair created
at two events (pass 3; R12 + R14 together).

THIS ROW IS RELATIVISTIC (1+1 dimensions, c = 1).  It combines the graph
postulates of M10 (``graph_frames.py``, R12) with the boost frames of M11
(``lorentz_frames.py``, R14) and adds a fourth postulate — light-cone
synchronisation at the transfer event — giving mechanism M12.

The problem.  In R14 both partners are created at one event, so "proper time
since creation" is an invariant clock with a common origin and the two
measurement events are ordered invariantly.  After a swap, A and D are
partners created at different events E_A, E_D (lab: (0, 0) and (0, x0)).
Each proper time is still invariant, but comparing them needs a convention
that relates two clocks on two worldlines with no common event — and the
frame time of M11 depends on where the pair frame is anchored (claim 1).

Postulate (iv).  Let T be the Bell-state-measurement (transfer) event.  The
clock zero of each new partner is the point where the FUTURE LIGHT CONE of
T crosses its worldline; the pair clock of a measurement is the proper time
from that point.  Light cones and proper times are invariant, so the
ordering is (claim 2); it reduces to R14 when T is the creation event
(claim 3); measurements before the crossing get negative values (claim 4);
lab-simultaneity synchronisation is not invariant (claim 5); a GHZ triple
born at one event is covered by R14 (claim 6); the delayed-choice geometry
of L-D2 gives negative clock values in every frame (claim 7).

Worldlines are (creation event, velocity); events are lab (t, x).
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from mapping_spaces.entangled import graph_frames as G, lorentz_frames as LZ, qm
from mapping_spaces.entangled import r12_swapping as r12, r14_lorentz_frames as r14

C_SI = 299_792_458.0  # m/s; only for the L-D2 geometry (claim 7)


# ------------------------------------------------------------ kinematics
def boost_event(u: float, ev) -> tuple[float, float]:
    """Lab event (t, x) as described by a lab moving at u."""
    t, x = ev
    gam = r14.gamma(u)
    return float(gam * (t - u * x)), float(gam * (x - u * t))


@dataclass(frozen=True)
class Worldline:
    """Inertial worldline through the creation event (tc, xc) at velocity v."""

    name: str
    z: float
    v: float
    tc: float = 0.0
    xc: float = 0.0

    def x(self, t):
        return self.xc + self.v * (np.asarray(t, float) - self.tc)

    def event(self, t) -> tuple[float, float]:
        return float(t), float(self.x(t))

    def proper_time(self, t) -> float:
        """Proper time since the creation event of the event at lab time t."""
        return float((t - self.tc) * np.sqrt(1.0 - self.v * self.v))

    def boosted(self, u: float) -> "Worldline":
        tc, xc = boost_event(u, (self.tc, self.xc))
        return Worldline(self.name, self.z, LZ.rel_velocity(self.v, u), tc, xc)

    def shifted_creation(self, delta: float) -> "Worldline":
        """Same worldline, creation reference moved by lab time delta."""
        return Worldline(self.name, self.z, self.v, self.tc + delta, float(self.x(self.tc + delta)))


def interval_proper_time(ev0, ev1) -> float:
    """Invariant proper time along the straight line from ev0 to ev1."""
    dt, dx = ev1[0] - ev0[0], ev1[1] - ev0[1]
    return float(np.sign(dt) * np.sqrt(max(dt * dt - dx * dx, 0.0)))


def light_cone_crossing(w: Worldline, T) -> float:
    """Lab time at which the future light cone of T crosses w.  With s the
    side of T on which w lies at t_T, t - t_T = s (x(t) - x_T) is linear in t."""
    tT, xT = T
    s = float(np.sign(w.x(tT) - xT))
    if s == 0.0:
        return float(tT)
    return float((tT + s * (w.xc - w.v * w.tc - xT)) / (1.0 - s * w.v))


def pair_clock(w: Worldline, T, t_meas: float) -> float:
    """Postulate (iv): proper time from the light-cone crossing to the
    measurement (negative before the crossing)."""
    return float(np.sqrt(1.0 - w.v * w.v) * (t_meas - light_cone_crossing(w, T)))


def lab_sync_clock(w: Worldline, T, t_meas: float) -> float:
    """Contrast rule: clock zero at lab time t_T on the worldline."""
    return float(np.sqrt(1.0 - w.v * w.v) * (t_meas - T[0]))


def in_future_cone(ev, T) -> bool:
    return (ev[0] - T[0]) >= abs(ev[1] - T[1])


def first_by(clocks: dict):
    """Name with the smallest clock value (None on a tie)."""
    items = sorted(clocks.items(), key=lambda kv: kv[1])
    if len(items) > 1 and abs(items[0][1] - items[1][1]) < 1e-15:
        return None
    return items[0][0]


# ------------------------------------------------------- configuration
HEIGHTS = {"A": -1.0, "B": 1.0, "C": -2.0, "D": 2.0}
VELOCITIES = {"A": -0.2, "B": 0.4, "C": -0.4, "D": 0.5}
X0 = 1.6                     # C, D created at (0, x0); A, B at (0, 0)
T_MEAS = {"A": 6.0, "D": 7.0}            # after both light-cone crossings
T_MEAS_EARLY = {"A": 3.0, "D": 4.0}      # after T, before the crossings


def swap_worldlines(delta: float = 0.0) -> dict:
    """A, B, C, D of the swap.  With delta != 0 the second source fires at
    lab time delta on D's worldline (same worldline for D; C's velocity is
    re-chosen so that it still meets B at the same transfer event)."""
    T = transfer_event()
    wl = {n: Worldline(n, HEIGHTS[n], VELOCITIES[n], 0.0, 0.0 if n in "AB" else X0) for n in "ABD"}
    if delta == 0.0:
        wl["C"] = Worldline("C", HEIGHTS["C"], VELOCITIES["C"], 0.0, X0)
    else:
        wl["D"] = wl["D"].shifted_creation(delta)
        vC = (T[1] - wl["D"].xc) / (T[0] - wl["D"].tc)
        if abs(vC) >= 1.0:
            raise ValueError("C would have to move faster than light")
        wl["C"] = Worldline("C", HEIGHTS["C"], float(vC), wl["D"].tc, wl["D"].xc)
    return wl


def transfer_event() -> tuple[float, float]:
    """Lab event where B and C meet (the BSM)."""
    tT = X0 / (VELOCITIES["B"] - VELOCITIES["C"])
    return float(tT), float(VELOCITIES["B"] * tT)


def graph_after_transfer(wl: dict) -> G.LocalityGraph:
    """The M10 graph after the BSM (A~D linked), as a LocalityGraph."""
    g = G.LocalityGraph()
    for n in "ABCD":
        g.add(G.Particle(n, wl[n].z, wl[n].v, float(wl[n].x(0.0))))
    g.link("A", "D", 1.0)
    return g


# ------------------------------------------------------------- claim 1
def m11_frame_times(wl: dict, anchor: str, t_meas: dict = T_MEAS) -> dict:
    """Frame-A times (M11 boost by psi_A(z)) of A's and D's measurement
    events, with the lab origin translated to the creation event of
    ``anchor``: frame time = tau_j - gamma_j v_j x_j(anchor)."""
    g = graph_after_transfer(wl)
    shift = wl[anchor].xc
    out = {}
    for n in ("A", "D"):
        t, x = wl[n].event(t_meas[n])
        t_, _ = LZ.to_frame(g, "A", wl[n].z, t, x - shift)
        out[n] = t_
    return out


def creation_clocks(wl: dict, t_meas: dict = T_MEAS) -> dict:
    return {n: wl[n].proper_time(t_meas[n]) for n in ("A", "D")}


def creation_clocks_under_boost(wl: dict, u: float, t_meas: dict = T_MEAS) -> dict:
    """The same measurement events described from a lab moving at u: proper
    times from the coordinate formula in the new lab."""
    out = {}
    for n in ("A", "D"):
        w_u = wl[n].boosted(u)
        t_u, _ = boost_event(u, wl[n].event(t_meas[n]))
        out[n] = w_u.proper_time(t_u)
    return out


def creation_order_vs_offset(deltas, t_meas: dict = T_MEAS) -> dict:
    """Order by 'proper time since own creation' when the second source
    fires at lab time delta (same D worldline, same transfer event)."""
    return {float(d): first_by(creation_clocks(swap_worldlines(d), t_meas)) for d in deltas}


# --------------------------------------------------------- claims 2–4
def pair_clocks(wl: dict, t_meas: dict = T_MEAS, rule=pair_clock) -> dict:
    T = transfer_event()
    return {n: rule(wl[n], T, t_meas[n]) for n in ("A", "D")}


def pair_clocks_under_boost(wl: dict, u: float, t_meas: dict = T_MEAS, rule=pair_clock) -> dict:
    """Transfer event, worldlines and measurement events all boosted; the
    clocks recomputed from scratch in the new lab."""
    T_u = boost_event(u, transfer_event())
    out = {}
    for n in ("A", "D"):
        w_u = wl[n].boosted(u)
        t_u, _ = boost_event(u, wl[n].event(t_meas[n]))
        out[n] = rule(w_u, T_u, t_u)
    return out


def crossing_times(wl: dict) -> dict:
    T = transfer_event()
    return {n: light_cone_crossing(wl[n], T) for n in ("A", "D")}


def light_travel_offsets(wl: dict) -> dict:
    """t_cross - t_T: by how much the invariant clock zero lags the
    instantaneous (lab-time t_T) graph edge on each worldline."""
    tT = transfer_event()[0]
    return {n: tc - tT for n, tc in crossing_times(wl).items()}


def r14_pair_as_worldlines(vA=-0.3, vB=0.8) -> dict:
    return {"A": Worldline("A", -1.0, vA), "B": Worldline("B", 1.0, vB)}


def clocks_when_transfer_is_creation(vA=-0.3, vB=0.8, tA=1.0, tB=1.2) -> dict:
    """Claim 3: T = the creation event (0, 0) gives crossing at T itself and
    the pair clocks equal the R14 proper times / frame times."""
    wl = r14_pair_as_worldlines(vA, vB)
    T = (0.0, 0.0)
    g = r14.pair(vA, vB)
    return {
        "crossing": {n: light_cone_crossing(wl[n], T) for n in wl},
        "clock": {"A": pair_clock(wl["A"], T, tA), "B": pair_clock(wl["B"], T, tB)},
        "r14_frame_times": dict(zip(("A", "B"), r14.frame_times(g, "A", "A", "B", tA, tB))),
        "r14_first": LZ.order_in_pair_frame(g, "A", "B", tA, tB),
    }


def swapped_pair_box(order: str):
    """Sequential box for the corrected swapped A–D state (BSM outcome 0),
    projecting ``order`` ('A' or 'D') first: returns (box, |S|, signalling,
    max deviation from the QM box)."""
    rho, _ = r12.conditional_ad(r12.initial_state(), r12.bsm_projector(0))
    rho = r12.correct_on_d(rho, 0)
    a, ap, b, bp = qm.chsh_optimal_settings()
    box = r14.sequential_box(rho, (a, ap), (b, bp), first="A" if order == "A" else "B")
    ref = qm.box_from_state(rho, (a, ap), (b, bp))
    return box, abs(qm.box_chsh(box)), qm.signalling(box), float(np.abs(box - ref).max())


# ------------------------------------------------------------- claim 6
GHZ_NAMES = ("P1", "P2", "P3")
GHZ_HEIGHTS = (-1.0, 1.0, 2.0)
GHZ_VELOCITIES = (-0.6, 0.6, 0.3)
GHZ_T_MEAS = (1.0, 0.9, 1.1)


def ghz_triple(velocities=GHZ_VELOCITIES) -> G.LocalityGraph:
    g = G.LocalityGraph()
    for n, z, v in zip(GHZ_NAMES, GHZ_HEIGHTS, velocities):
        g.add(G.Particle(n, z, v))
    for i, j in itertools.combinations(GHZ_NAMES, 2):
        g.link(i, j, 1.0)
    return g


def ghz_frame_events(g: G.LocalityGraph, k: str, t_meas=GHZ_T_MEAS) -> dict:
    """Frame-k (t', x') of the three measurement events."""
    out = {}
    for n, t in zip(GHZ_NAMES, t_meas):
        p = g.particles[n]
        out[n] = LZ.to_frame(g, k, p.z, t, p.v * t)
    return out


def ghz_order_under_boost(u: float, t_meas=GHZ_T_MEAS) -> tuple:
    """Proper-time order of the three events described from a lab at u."""
    g = ghz_triple()
    h = LZ.boosted_lab(g, u)
    gam = r14.gamma(u)
    times, xs = {}, {}
    for n, t in zip(GHZ_NAMES, t_meas):
        p, p_u = g.particles[n], h.particles[n]
        t_u = gam * t * (1.0 - u * p.v)          # origin-anchored worldline
        times[n], xs[n] = LZ.to_frame(h, "P1", p.z, t_u, p_u.v * t_u)
    return tuple(sorted(times, key=times.get)), times, xs


def ghz_light_cone_clocks(t_meas=GHZ_T_MEAS) -> dict:
    """Postulate (iv) with T = the common creation event: equals tau."""
    return {n: pair_clock(Worldline(n, z, v), (0.0, 0.0), t)
            for n, z, v, t in zip(GHZ_NAMES, GHZ_HEIGHTS, GHZ_VELOCITIES, t_meas)}


# ------------------------------------------------------------- claim 7
LD2_DELAY_NS = 485.0   # Victor's BSM after Alice's and Bob's measurements (L-D2)


def ld2_clocks(d_metres, betas=(0.0, 1e-3, -1e-3), delay_ns: float = LD2_DELAY_NS) -> dict:
    """L-D2 geometry with massive stand-ins at rest: Alice's qubit at x = -d,
    Bob's at x = +d (light-ns), measured at lab t = 0; Victor's BSM at
    T = (delay, 0).  Returns the pair clocks (ns) and whether the measurement
    lies in T's future cone, for each global boost beta."""
    d = float(d_metres) / C_SI * 1e9      # light-ns
    T = (delay_ns, 0.0)
    wl = {"A": Worldline("A", -1.0, 0.0, -1e3, -d), "D": Worldline("D", 2.0, 0.0, -1e3, d)}
    out = {}
    for beta in betas:
        row = {}
        T_u = boost_event(beta, T)
        for n in ("A", "D"):
            w_u = wl[n].boosted(beta)
            ev_u = boost_event(beta, wl[n].event(0.0))
            row[n] = pair_clock(w_u, T_u, ev_u[0])
            row[n + "_in_future_cone"] = in_future_cone(ev_u, T_u)
        out[float(beta)] = row
    return out


def ld2_cone_boundary_metres(delay_ns: float = LD2_DELAY_NS) -> float:
    """Distance at which Alice's event would be lightlike to T (derived)."""
    return float(delay_ns * 1e-9 * C_SI)


if __name__ == "__main__":
    wl = swap_worldlines()
    print("transfer event:", transfer_event())
    print("claim 1 frame times, anchor A / D:", m11_frame_times(wl, "A"), m11_frame_times(wl, "D"))
    print("claim 1 creation clocks:", creation_clocks(wl), {u: creation_clocks_under_boost(wl, u) for u in (-0.5, 0.2, 0.9)})
    print("claim 1 order vs source offset:", creation_order_vs_offset((-0.5, 0.0, 0.5)))
    print("claim 2 crossings:", crossing_times(wl), "clocks:", pair_clocks(wl))
    print("claim 2 boosted:", {u: pair_clocks_under_boost(wl, u) for u in (-0.5, 0.2, 0.9)})
    print("claim 3:", clocks_when_transfer_is_creation())
    print("claim 4 early clocks:", pair_clocks(wl, T_MEAS_EARLY), "offsets:", light_travel_offsets(wl))
    print("claim 4 boxes:", {o: swapped_pair_box(o)[1:] for o in ("A", "D")})
    print("claim 5 lab-sync:", pair_clocks(wl, rule=lab_sync_clock),
          {u: pair_clocks_under_boost(wl, u, rule=lab_sync_clock) for u in (-0.5, 0.0, 0.5)})
    print("claim 6 GHZ:", {u: ghz_order_under_boost(u) for u in (0.0, -0.5, 0.9)}, ghz_light_cone_clocks())
    print("claim 7 L-D2:", {d: ld2_clocks(d) for d in (1.0, 10.0, 100.0)}, ld2_cone_boundary_metres())


# ------------------------------------------------- round-5 item H4 (B-N3)
def chart_jump_at_reanchoring() -> dict:
    """The covariant chart applies the boost Lambda(psi(z)) from the
    laboratory origin, so when the transfer's cone reaches a particle its
    chart re-pins and positions jump.  In the working pair the cone of the
    transfer T = (2, 0.8) reaches A's worldline at t = 3.5; at that event A's
    chart moves its former partner B from 0 to 2.14 and its new partner D
    from 4.13 to 1.85 (= gamma(0.5) * 1.6).  Returned: positions before and
    after in A's chart, at the crossing event."""
    from mapping_spaces.entangled import lorentz_frames as LF
    wl = swap_worldlines()
    T = transfer_event()
    tA = light_cone_crossing(wl["A"], T)

    def graph(links):
        g = G.LocalityGraph()
        for n in "ABCD":
            g.add(G.Particle(n, wl[n].z, wl[n].v, float(wl[n].x(0.0))))
        for a, b in links:
            g.link(a, b, 1.0)
        return g
    before, after = graph([("A", "B"), ("C", "D")]), graph([("A", "D")])
    out = {"t_crossing": float(tA)}
    for who in ("B", "D"):
        xb = LF.to_frame(before, "A", wl[who].z, tA, wl[who].x(tA))[1]
        xa = LF.to_frame(after, "A", wl[who].z, tA, wl[who].x(tA))[1]
        out[who] = {"before": float(xb), "after": float(xa)}
    return out
