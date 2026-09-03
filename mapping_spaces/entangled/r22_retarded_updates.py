"""R22 — the retarded transfer postulate and the chart-locality theorem
(paper-1 revision, handoff B5/D10 and C2).

Pre-revision, the covariant variant was a hybrid: the swapped edge A-D was
created on the lab slice t = t_T of the transfer event T while the pair clock
was zeroed on T's future light cone — so between slice and cone the edge's
existence was frame-dependent.  The revision adopts the *retarded* reading:

    every graph update caused by an event E (edge creation at an entangling
    or transfer event, edge removal at a measurement) reaches particle i's
    chart where the future light cone of E meets i's worldline.

Consequences, each verified here:

* edge existence becomes an invariant: "the edge to i exists at event ev" is
  "ev lies in the future light cone of E", a Lorentz-invariant relation
  (r16.in_future_cone; the sign of the pair clock kappa).
* the chart-locality theorem (Proposition 3 of the revised paper): no frame
  potential changes outside the future light cone of the event that causes the
  change — true by construction under the retarded reading, and the
  construction is frame-independent because the cone-crossing of a worldline
  is equivariant under boosts (verified numerically here).
* the statistics are untouched: potentials never act on the spin state
  (Proposition 1(iii)), and the ordering rule (P4) already used the cone.
* in the delayed-choice geometry both outer measurements lie outside the
  future cone of the transfer in every frame (kappa < 0 invariantly), so under
  the retarded reading no A-D edge ever exists at the outer measurements:
  the delayed-choice cost is stated with no hybrid left over.
"""
from __future__ import annotations

import numpy as np

from . import r16_covariant_swapping as r16


def edge_exists(ev, T) -> bool:
    """Retarded reading: the update caused at T has reached event ev."""
    return r16.in_future_cone(ev, T)


def cone_crossing_equivariant(v: float, x0: float, t0: float, T, u: float,
                              tol: float = 1e-9) -> bool:
    """The event where particle i's chart changes (worldline meets T's future
    cone) transforms as an event: boosting the configuration and recomputing
    gives the boost of the original crossing."""
    w = r16.Worldline(name="i", z=1.0, v=v, tc=t0, xc=x0)
    tc = r16.light_cone_crossing(w, T)
    ev = w.event(tc)
    wb = w.boosted(u)
    Tb = r16.boost_event(u, T)
    tcb = r16.light_cone_crossing(wb, Tb)
    evb = wb.event(tcb)
    ref = r16.boost_event(u, ev)
    return abs(evb[0] - ref[0]) < tol and abs(evb[1] - ref[1]) < tol


def edge_existence_invariant(ev, T, us) -> bool:
    """Membership of the future cone is the same in every boosted frame."""
    base = edge_exists(ev, T)
    return all(edge_exists(r16.boost_event(u, ev), r16.boost_event(u, T)) == base
               for u in us)


def random_configurations(m: int = 500, seed: int = 5):
    rng = np.random.default_rng(seed)
    for _ in range(m):
        T = (rng.uniform(-2, 2), rng.uniform(-2, 2))
        ev = (rng.uniform(-4, 4), rng.uniform(-4, 4))
        u = rng.uniform(-0.9, 0.9)
        yield ev, T, u


def delayed_choice_retarded(ds=(1.0, 10.0, 100.0), us=(0.0, 1e-3, -1e-3)) -> dict:
    """The Ma-geometry stand-ins (r16 conventions, seconds/light-seconds):
    outer particles at rest at x = -/+d measured at t = 0, transfer at
    t = 485 ns.  Under the retarded reading no A-D edge exists at either outer
    measurement, in any frame."""
    c = 299792458.0
    out = {}
    for d_m in ds:
        d = d_m / c
        T = (485e-9, 0.0)
        for label, x in (("A", -d), ("D", d)):
            ev = (0.0, x)
            out[(d_m, label)] = {
                "edge_at_measurement": edge_exists(ev, T),
                "invariant": edge_existence_invariant(ev, T, us),
            }
    return out
