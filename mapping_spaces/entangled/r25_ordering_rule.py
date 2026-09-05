"""R25 — the repaired covariant ordering rule (round-4 revision, items B1/B2).

Round 4 refuted the light-cone clock of Eq. (10) in two ways (referees B and C,
independently): (i) for a transfer-anchored pair the two clocks start at two
different cone crossings, and the proper time from the crossing to the
measurement can order two *timelike*-separated measurements against their
causal order (3.6-4.8 % of random timelike pairs); (ii) P4's per-worldline
anchor ("most recent crossing of its own worldline") compares clocks zeroed at
different events and inverts a causal order in a GHZ-pruning configuration.

The repaired rule, implemented and verified here (1+1 dimensions, c = 1;
events are (t, x)):

  * The ordering clock of a component is the SIGNED INVARIANT INTERVAL from
    the component's defining event E (creation, transfer or the pruning
    measurement) to the measurement event e,
        kappa(e) = +sqrt(dt^2 - dx^2)   if e is in the future cone of E,
                   -sqrt(dx^2 - dt^2)   if e is spacelike to E,
                   -sqrt(dt^2 - dx^2)   if e is in the past cone of E,
    for EVERY member of the component (one anchor per component, no
    per-worldline anchor).  In words: the proper time of an inertial clock
    carried from the defining event to the measurement event.  Its level
    sets in E's future cone are the Milne hyperboloids of that cone.  For a
    creation-anchored component (worldlines through E) it is the proper time
    since creation, i.e. the original creation rule.
  * The order of the component's measurement events is the transitive closure
    of {causal order where relativity defines one} together with {kappa-order
    among the events lying in E's future cone}.  Events outside that cone
    (where, under retarded P2, the component's edges have not yet reached the
    chart) are ordered by the causal order alone.
  * Theorem (verified below).  kappa is Lorentz invariant and, inside E's
    future cone, strictly increasing along every future-directed causal curve
    (reverse triangle inequality), so the kappa-order there agrees with the
    causal order wherever the latter is defined; the closure is therefore
    acyclic, a strict partial order, invariant under boosts, and total (up
    to ties) on the in-cone events.  No Lorentz-invariant clock can do the
    same outside the cone: on the spacelike region every invariant level set
    contains causally ordered pairs (demonstrated numerically).
  * The ordering clock is invariant but NOT retarded: kappa at an event may
    refer to a defining event outside that event's past cone.  Graph (P2,
    retarded) and clock (P4, invariant) are different objects.

The two referee counterexamples are reproduced under the old rule and defeated
under the new one; random configurations with creation, transfer and pruning
anchors, timelike and spacelike pairs, in-cone and out-of-cone events, GHZ
triples at three mutually spacelike events, and random boosts are checked for
causal consistency, transitivity and invariance.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from mapping_spaces.entangled import r16_covariant_swapping as r16

Event = tuple[float, float]


# ------------------------------------------------------------------ kinematics
def boost_event(u: float, ev) -> Event:
    return r16.boost_event(u, ev)


def causal(e1, e2, tol: float = 1e-12) -> int:
    """+1 if e2 lies in the closed future cone of e1 (e1 before e2), -1 if the
    reverse, 0 if spacelike-separated (or identical)."""
    dt, dx = e2[0] - e1[0], e2[1] - e1[1]
    if dt >= abs(dx) - tol and dt > tol:
        return 1
    if -dt >= abs(dx) - tol and -dt > tol:
        return -1
    return 0


def in_future_cone(e, E, tol: float = 1e-12) -> bool:
    return causal(E, e, tol) == 1 or (abs(e[0] - E[0]) < tol and abs(e[1] - E[1]) < tol)


def signed_interval(E, e) -> float:
    """The ordering clock kappa(e) of the component defined at E."""
    dt, dx = e[0] - E[0], e[1] - E[1]
    s2 = dt * dt - dx * dx
    if s2 >= 0.0:
        return float(np.sign(dt) * np.sqrt(s2))
    return float(-np.sqrt(-s2))


# ------------------------------------------------------------- the ordering
def _closure(R: np.ndarray) -> np.ndarray:
    R = R.copy()
    n = len(R)
    for k in range(n):
        R |= R[:, k][:, None] & R[k, :][None, :]
    return R


def order_relation(events, E, tol: float = 1e-12) -> np.ndarray:
    """Boolean matrix R with R[i, j] = True iff event i precedes event j under
    the repaired rule (transitive closure of causal order and in-cone
    kappa-order).  Ties (equal kappa, spacelike) are left unordered."""
    n = len(events)
    R = np.zeros((n, n), dtype=bool)
    kap = [signed_interval(E, e) for e in events]
    inside = [in_future_cone(e, E, tol) for e in events]
    for i, j in itertools.permutations(range(n), 2):
        c = causal(events[i], events[j], tol)
        if c == 1:
            R[i, j] = True
        elif c == 0 and inside[i] and inside[j] and kap[i] < kap[j] - tol:
            R[i, j] = True
    return _closure(R)


def is_strict_partial_order(R: np.ndarray) -> bool:
    n = len(R)
    irreflexive = not R.diagonal().any()
    antisymmetric = not (R & R.T).any()
    transitive = bool(np.array_equal(_closure(R), R))
    return irreflexive and antisymmetric and transitive and n >= 0


def consistent_with_causal(R: np.ndarray, events, tol: float = 1e-12) -> bool:
    for i, j in itertools.permutations(range(len(events)), 2):
        if causal(events[i], events[j], tol) == 1 and not R[i, j]:
            return False
        if causal(events[i], events[j], tol) == 1 and R[j, i]:
            return False
    return True


def first(events, E):
    """For a pair: index of the earlier event, or None if unordered."""
    R = order_relation(events, E)
    if R[0, 1]:
        return 0
    if R[1, 0]:
        return 1
    return None


def boosted(events, E, u: float):
    return [boost_event(u, e) for e in events], boost_event(u, E)


# ------------------------------------------------------- the theorem, numerically
def kappa_monotone_in_cone(n: int = 20000, seed: int = 25) -> dict:
    """Inside E's future cone kappa increases along every causal step: for
    random in-cone e1 and random e2 in e1's future cone, kappa(e2) > kappa(e1),
    and the gap is at least the proper time of the step (reverse triangle
    inequality)."""
    rng = np.random.default_rng(seed)
    E = (0.0, 0.0)
    worst = np.inf
    count = 0
    for _ in range(n):
        t1 = rng.uniform(0.1, 5.0)
        e1 = (t1, rng.uniform(-t1, t1))
        dt = rng.uniform(0.0, 5.0)
        e2 = (e1[0] + dt, e1[1] + rng.uniform(-dt, dt))
        k1, k2 = signed_interval(E, e1), signed_interval(E, e2)
        step = r16.interval_proper_time(e1, e2)
        worst = min(worst, (k2 - k1) - step)
        count += 1
    return {"pairs": count, "min_excess_over_step": float(worst), "monotone": worst >= -1e-9}


def no_invariant_time_function_outside_cone() -> dict:
    """Two events on the same invariant level set (spacelike distance 5 from
    E) that are nevertheless causally ordered: no Lorentz-invariant function
    of the displacement from E is strictly monotone along causal curves in the
    spacelike region."""
    E = (0.0, 0.0)
    e1 = (0.0, 5.0)
    t = 30.0
    e2 = (t, float(np.sqrt(25.0 + t * t)))
    return {"kappa_1": signed_interval(E, e1), "kappa_2": signed_interval(E, e2),
            "causal_1_before_2": causal(e1, e2) == 1}


# ------------------------------------------------- referee counterexamples
def _old_pair_clock(x0, t0, v, T, t_meas) -> float:
    """Eq. (10) of v4: proper time along the worldline from the crossing of
    T's future cone to the measurement (the refuted rule)."""
    w = r16.Worldline("w", 1.0, v, t0, x0)
    return r16.pair_clock(w, T, t_meas)


def referee_B_ghz_pruning() -> dict:
    """Referee B's configuration (check_p4_anchor.py): GHZ triple created at O;
    particle 1 (v = -0.5) measured at E1 = (1, -0.5); particle 2 (v = 0) at
    E2 = (1.4, 0), outside E1's cone; particle 3 (v = 0.5) at E3 = (3.1, 1.55),
    inside E1's cone and in E2's causal future.  v4's per-worldline anchor
    ordered 3 before 2; the repaired rule (both anchored at the pruning event
    E1) orders 2 before 3, the causal order."""
    E1, E2, E3 = (1.0, -0.5), (1.4, 0.0), (3.1, 1.55)
    # v4 mixed anchoring: 2 since creation O, 3 since E1's cone crossing
    k2_old = _old_pair_clock(0.0, 0.0, 0.0, (0.0, 0.0), 1.4)
    k3_old = _old_pair_clock(0.0, 0.0, 0.5, E1, 3.1)
    R = order_relation([E2, E3], E1)
    return {"causal": causal(E2, E3), "old_order": "3 first" if k3_old < k2_old else "2 first",
            "old_clocks": (k2_old, k3_old),
            "new_clocks": (signed_interval(E1, E2), signed_interval(E1, E3)),
            "new_order": "2 first" if R[0, 1] else ("3 first" if R[1, 0] else "unordered"),
            "new_agrees_with_causal": bool(R[0, 1])}


def referee_C_instance() -> dict:
    """Referee C's instance (check_C_kappa_causal.py, Part A): transfer at
    T = (0, 0); A at rest at x = 0.1 measured at M_A = (2, 0.1); D moving at
    0.9c from x = -22.9 at t = -23, measured at M_D = (3, 0.5).  M_D is in
    M_A's causal future; Eq. (10) gave kappa_A = 1.900, kappa_D = 0.803 (D
    'first'); the repaired rule orders A first."""
    T = (0.0, 0.0)
    MA, MD = (2.0, 0.1), (3.0, 0.5)
    kA_old = _old_pair_clock(0.1, 0.0, 0.0, T, 2.0)
    kD_old = _old_pair_clock(-22.9, -23.0, 0.9, T, 3.0)
    R = order_relation([MA, MD], T)
    return {"causal": causal(MA, MD), "old_clocks": (kA_old, kD_old),
            "old_order": "D first" if kD_old < kA_old else "A first",
            "new_clocks": (signed_interval(T, MA), signed_interval(T, MD)),
            "new_order": "A first" if R[0, 1] else ("D first" if R[1, 0] else "unordered"),
            "new_agrees_with_causal": bool(R[0, 1])}


def referee_B_worked_instance() -> dict:
    """Referee B's worked instance: particle 1 on x = 2.982 - 0.870 t measured
    at t = 5.684, particle 2 on x = -0.053 - 0.244 t at t = 4.930, transfer at
    the origin; event 2 is in event 1's past cone, yet Eq. (10) gave
    kappa_1 = 2.014 < kappa_2 = 4.712."""
    T = (0.0, 0.0)
    e1 = (5.684, 2.982 - 0.870 * 5.684)
    e2 = (4.930, -0.053 - 0.244 * 4.930)
    k1_old = _old_pair_clock(2.982, 0.0, -0.870, T, 5.684)
    k2_old = _old_pair_clock(-0.053, 0.0, -0.244, T, 4.930)
    R = order_relation([e1, e2], T)
    return {"causal_2_before_1": causal(e2, e1) == 1, "old_clocks": (k1_old, k2_old),
            "old_inverted": k1_old < k2_old,
            "new_clocks": (signed_interval(T, e1), signed_interval(T, e2)),
            "new_2_before_1": bool(R[1, 0])}


# ------------------------------------------------------ random configurations
def random_transfer_pairs(n: int = 200000, seed: int = 21) -> dict:
    """Referee B's sampling: transfer at the origin, two worldlines with random
    positions at t = 0 and velocities, random measurement times, keeping the
    pairs with both measurements inside the transfer's future cone.  Counts
    timelike pairs whose Eq. (10) order is inverted, and the same under the
    repaired rule."""
    rng = np.random.default_rng(seed)
    T = (0.0, 0.0)
    timelike = inverted_old = inverted_new = 0
    for _ in range(n):
        x0 = rng.uniform(-3, 3, 2)
        vv = rng.uniform(-0.95, 0.95, 2)
        tm = rng.uniform(0, 8, 2)
        ev = [(float(tm[i]), float(x0[i] + vv[i] * tm[i])) for i in range(2)]
        if not (in_future_cone(ev[0], T) and in_future_cone(ev[1], T)):
            continue
        c = causal(ev[0], ev[1])
        if c == 0:
            continue
        timelike += 1
        k_old = [_old_pair_clock(x0[i], 0.0, vv[i], T, tm[i]) for i in range(2)]
        if (k_old[0] < k_old[1]) != (c == 1):
            inverted_old += 1
        R = order_relation(ev, T)
        if (c == 1 and not R[0, 1]) or (c == -1 and not R[1, 0]):
            inverted_new += 1
    return {"timelike_pairs": timelike, "inverted_old": inverted_old,
            "inverted_new": inverted_new,
            "fraction_old": inverted_old / max(timelike, 1)}


def random_components(n: int = 3000, members: int = 4, seed: int = 25,
                      boosts=(-0.6, 0.3, 0.8)) -> dict:
    """Random components of `members` measurement events with a random
    defining event E (off every worldline in general), events drawn inside
    and outside E's future cone (mixed anchors: some in the cone of the
    defining event, some before it): the rule is a strict partial order,
    consistent with the causal order, and the relation matrix is the same in
    every boosted frame."""
    rng = np.random.default_rng(seed)
    bad_order = bad_causal = bad_invariant = 0
    in_cone_events = 0
    for _ in range(n):
        E = (float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)))
        events = []
        for _m in range(members):
            if rng.uniform() < 0.6:            # inside the cone
                dt = rng.uniform(0.05, 4.0)
                events.append((E[0] + dt, E[1] + rng.uniform(-dt, dt)))
                in_cone_events += 1
            else:                              # anywhere
                events.append((E[0] + rng.uniform(-4, 4), E[1] + rng.uniform(-4, 4)))
        R = order_relation(events, E)
        if not is_strict_partial_order(R):
            bad_order += 1
        if not consistent_with_causal(R, events):
            bad_causal += 1
        for u in boosts:
            evb, Eb = boosted(events, E, u)
            if not np.array_equal(order_relation(evb, Eb, tol=1e-9), R):
                bad_invariant += 1
                break
    return {"components": n, "members": members, "in_cone_events": in_cone_events,
            "not_partial_order": bad_order, "causal_violations": bad_causal,
            "not_invariant": bad_invariant}


def ghz_spacelike_triple(velocities=(-0.6, 0.0, 0.6), t_meas=(1.0, 1.05, 1.0),
                         boosts=(-0.6, 0.3, 0.8)) -> dict:
    """A GHZ triple created at the origin and measured at three mutually
    spacelike events: the rule is the proper-time order (creation anchor), a
    total order, transitive and boost-invariant."""
    E = (0.0, 0.0)
    events = [(t, v * t) for v, t in zip(velocities, t_meas)]
    pairs = [causal(events[i], events[j]) for i, j in itertools.combinations(range(3), 2)]
    R = order_relation(events, E)
    kap = [signed_interval(E, e) for e in events]
    tau = [t * np.sqrt(1 - v * v) for v, t in zip(velocities, t_meas)]
    order = tuple(int(i) for i in np.argsort(kap))
    invariant = all(np.array_equal(order_relation(*boosted(events, E, u), tol=1e-9), R)
                    for u in boosts)
    total = all(R[i, j] or R[j, i] for i, j in itertools.combinations(range(3), 2))
    return {"mutually_spacelike": all(c == 0 for c in pairs), "kappa": kap,
            "proper_times": tau, "order": order, "total": total,
            "strict_partial_order": is_strict_partial_order(R), "invariant": invariant}


def creation_rule_reduction(n: int = 500, seed: int = 3) -> float:
    """For worldlines through E, kappa equals the proper time since E (the
    creation rule of Sec. 3.7.1 and Eq. (10) with T = E): max deviation."""
    rng = np.random.default_rng(seed)
    E = (0.0, 0.0)
    worst = 0.0
    for _ in range(n):
        v, t = rng.uniform(-0.95, 0.95), rng.uniform(0.01, 5)
        worst = max(worst, abs(signed_interval(E, (t, v * t)) - t * np.sqrt(1 - v * v)))
    return float(worst)


# ------------------------------------------------- the paper's worked numbers
def working_pair_clocks() -> dict:
    """The swapped pair of Sec. 3.7.2 (creation at (0,0) and (0,1.6),
    velocities (-0.2, 0.4, -0.4, 0.5), transfer at T = (2, 0.8)) measured at
    t_A = 6, t_D = 7 and, early, at t_A = 3, t_D = 4."""
    wl = r16.swap_worldlines()
    T = r16.transfer_event()
    out = {}
    for label, tm in (("late", r16.T_MEAS), ("early", r16.T_MEAS_EARLY)):
        events = [wl["A"].event(tm["A"]), wl["D"].event(tm["D"])]
        R = order_relation(events, T)
        k = {n: signed_interval(T, e) for n, e in zip(("A", "D"), events)}
        k_old = {n: r16.pair_clock(wl[n], T, tm[n]) for n in ("A", "D")}
        inv = all(abs(signed_interval(boost_event(u, T), boost_event(u, e)) - k[n]) < 1e-9
                  for u in (-0.5, 0.2, 0.9) for n, e in zip(("A", "D"), events))
        out[label] = {"kappa": k, "kappa_eq10": k_old,
                      "first": "A" if R[0, 1] else ("D" if R[1, 0] else None),
                      "in_cone": {n: in_future_cone(e, T) for n, e in zip(("A", "D"), events)},
                      "spacelike": causal(*events) == 0, "invariant": inv}
    return out


def delayed_choice_clocks(d_metres, betas=(0.0, 1e-3, -1e-3), delay_ns: float = r16.LD2_DELAY_NS) -> dict:
    """The delayed-choice geometry of Sec. 3.7.2 (massive stand-ins at rest at
    x = -+d measured at t = 0, transfer at T = (delay, 0)): both outer events
    lie outside T's future cone in every frame (kappa < 0), and are unordered
    by the rule (spacelike to each other, outside the cone)."""
    d = float(d_metres) / r16.C_SI * 1e9
    T = (delay_ns, 0.0)
    events = [(0.0, -d), (0.0, d)]
    out = {}
    for beta in betas:
        evb, Tb = boosted(events, T, beta)
        R = order_relation(evb, Tb, tol=1e-6)
        out[float(beta)] = {"kappa": tuple(signed_interval(Tb, e) for e in evb),
                            "in_cone": tuple(in_future_cone(e, Tb, tol=1e-6) for e in evb),
                            "ordered": bool(R[0, 1] or R[1, 0])}
    return out


def nonrelativistic_window(vA: float, vB: float, c: float = r16.C_SI, track_error: float = 0.01) -> dict:
    """Creation-anchored pair measured at lab times t_A, t_B: the kappa-order
    (proper times) differs from the lab order only for
    1 < t_B/t_A < gamma_B/gamma_A, a relative window of width
    ~ (v_B^2 - v_A^2)/2c^2; a relative track-velocity error `track_error`
    shifts the boundary by ~ (v_A^2 + v_B^2) track_error / c^2, inside which
    the order is undetermined."""
    gA, gB = 1 / np.sqrt(1 - (vA / c) ** 2), 1 / np.sqrt(1 - (vB / c) ** 2)
    width = abs(gB / gA - 1.0)
    blur = (vA * vA + vB * vB) * track_error / c**2
    return {"window": float(width), "approx": float(abs(vB * vB - vA * vA) / (2 * c * c)),
            "undetermined": float(blur)}


if __name__ == "__main__":
    print("monotone in cone:", kappa_monotone_in_cone())
    print("no invariant time function outside:", no_invariant_time_function_outside_cone())
    print("referee B GHZ pruning:", referee_B_ghz_pruning())
    print("referee C instance:", referee_C_instance())
    print("referee B worked instance:", referee_B_worked_instance())
    print("random transfer pairs:", random_transfer_pairs(50000))
    print("random components:", random_components(500))
    print("GHZ spacelike triple:", ghz_spacelike_triple())
    print("creation reduction:", creation_rule_reduction())
    print("working pair:", working_pair_clocks())
    for d in (1.0, 10.0, 100.0):
        print("delayed choice", d, "m:", delayed_choice_clocks(d))
    print("NR window:", nonrelativistic_window(1e4, 2e4))
