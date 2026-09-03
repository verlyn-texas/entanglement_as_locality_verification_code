"""R13 — three particles: GHZ locality on a one-dimensional qD, and what the
locality graph does after a measurement on one of them.

Claims (numbered as in rows/R13_ghz.md):
 1. Three particles created at one lab event, placed at z = -1, 1, 2 with
    three different lab velocities and all pairwise weights w = 1, are
    mutually at rest in each other's frames (interpolant degree 3, one qD
    dimension); in each particle's frame the other two sit at x = 0 for all
    y, while the lab sees three tracks fanning out.
 2. Spin sector: GHZ Mermin value 4 (LHV bound 2, R11); the all-versus-
    nothing signs <XXX> = +1, <XYY> = <YXY> = <YYX> = -1, which none of the 64
    local-realist assignments satisfies (at most 3 of the 4).  Every pairwise
    reduced state has concurrence 0, so a pairwise weight w = C (R05) would
    give an edgeless graph for a jointly entangled triple: the graph must be
    defined by *entangled-component membership* (plan §10.4 Q1 default).
 3. Measurement on particle 1 (state decides the graph): along Z the
    remaining pair is a product state (C_23 = 0) — all three edges go,
    including 2–3, although 2 and 3 were not measured; along X the pair is a
    Bell state (C_23 = 1) — edge 2–3 persists.  Along a direction at polar
    angle theta from z: C_23 = |sin theta|.  Postulate (ii) alone (remove
    the measured particle's edges) keeps 2–3 in the Z case: the state-driven
    rule is needed.
 4. Frame positions are continuous through the measurement event; after it,
    the separation of 2 and 3 in frame 2 grows as (v3 - v2)(y - y_m) in the
    Z case and stays 0 in the X case; the lab tracks never change.
 5. Visibility scaling: for rho_V = V|GHZ><GHZ| + (1 - V) I/8 the Mermin value
    is 4V; > 2 needs V > 1/2; V = 0.71 (L-D5a) gives 2.84; 2.77 (L-D5b)
    corresponds to V = 0.69; a fraction f = 0.85 of GHZ-conforming events
    (L-D5a) gives correlators 2f - 1 = 0.70 and Mermin 2.80.
 6. N particles at N distinct heights are mutually local on a 1-D qD for
    N = 3, 6, 14 (interpolant degree N); no second qD dimension is needed.
"""
from __future__ import annotations

import itertools
import numpy as np

from mapping_spaces.entangled import graph_frames as G, qm
from mapping_spaces.entangled import r11_locality_graph as r11

NAMES = ("P1", "P2", "P3")
HEIGHTS = (-1.0, 1.0, 2.0)
VELOCITIES = (-1.0, 1.0, 0.5)


# ------------------------------------------------------------ n-qubit tools
def reduced_n(rho: np.ndarray, keep, n: int) -> np.ndarray:
    """Partial trace of an n-qubit density matrix onto the qubits ``keep``
    (a sorted tuple of indices, particle 0 first)."""
    keep = tuple(keep)
    t = np.asarray(rho, dtype=complex).reshape((2,) * (2 * n))
    for q in sorted(set(range(n)) - set(keep), reverse=True):
        # trace out qubit q: axes q (row) and current-n + q (column)
        cur = t.ndim // 2
        t = np.trace(t, axis1=q, axis2=cur + q)
    m = 2 ** len(keep)
    return t.reshape(m, m)


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def project_particle(state, n_dir, outcome: int, which: int, n: int):
    """qm.project extended to n qubits: projective measurement of n_dir.sigma
    on qubit ``which`` (0-based).  Returns (post-measurement density matrix,
    probability)."""
    rho = qm.as_dm(state)
    ops = [qm.I2] * n
    ops[which] = qm.projector(n_dir, outcome)
    P = qm.kron(*ops)
    prob = float(np.real(np.trace(P @ rho)))
    if prob == 0:
        raise ValueError("outcome has zero probability")
    return P @ rho @ P / prob, prob


def entangled_components(state, n: int, tol: float = 1e-9):
    """Finest product decomposition of a *pure* n-qubit state: the component
    of particle i is the smallest subset S containing i whose reduced state
    is pure (|psi> = |phi_S> (x) |phi_rest>).  Returns a list of sorted
    tuples."""
    rho = qm.as_dm(state)
    if abs(purity(rho) - 1.0) > tol:
        raise ValueError("component rule implemented for pure states only")
    remaining = set(range(n))
    comps = []
    while remaining:
        i = min(remaining)
        others = sorted(remaining - {i})
        found = None
        for size in range(0, len(others) + 1):
            for extra in itertools.combinations(others, size):
                S = tuple(sorted((i, *extra)))
                if abs(purity(reduced_n(rho, S, n)) - 1.0) < tol:
                    found = S
                    break
            if found:
                break
        comps.append(found)
        remaining -= set(found)
    return comps


# ------------------------------------------------------------------ claim 1
def triple(heights=HEIGHTS, velocities=VELOCITIES, names=NAMES) -> G.LocalityGraph:
    """Three particles born at x = 0, y = 0, all pairwise local."""
    g = G.LocalityGraph()
    for name, z, v in zip(names, heights, velocities):
        g.add(G.Particle(name, z, v))
    for i, j in itertools.combinations(names, 2):
        g.link(i, j, 1.0)
    return g


def velocity_table(g: G.LocalityGraph) -> dict:
    """V[frame][particle-or-lab] = velocity seen in that frame."""
    frames = list(g.particles) + [G.LAB]
    return {k: {j: g.velocity_in_frame(j, k) for j in frames} for k in frames}


def positions_in_frames(g: G.LocalityGraph, y_grid) -> dict:
    """x^(k)_j(y) for every frame k and particle j (no events)."""
    h = G.History(g)
    return {k: {j: h.position_in_frame(j, k, y_grid) for j in g.particles}
            for k in list(g.particles) + [G.LAB]}


# ------------------------------------------------------------------ claim 2
def ghz_signs() -> dict:
    """<XXX>, <XYY>, <YXY>, <YYX> for the GHZ state (R11's sign convention)."""
    psi = r11.ghz()
    X, Y = qm.SX, qm.SY
    ops = {"XXX": qm.kron(X, X, X), "XYY": qm.kron(X, Y, Y),
           "YXY": qm.kron(Y, X, Y), "YYX": qm.kron(Y, Y, X)}
    return {k: float(np.real(psi.conj() @ op @ psi)) for k, op in ops.items()}


def lhv_assignments_satisfying(signs: dict | None = None):
    """Over the 64 deterministic assignments (x_i, y_i) in {+-1}^6, count how
    many of the four GHZ sign constraints each satisfies.  Returns
    (number satisfying all four, maximum satisfied by any assignment)."""
    signs = {k: int(round(v)) for k, v in (signs or ghz_signs()).items()}
    n_all, best = 0, 0
    for x1, y1, x2, y2, x3, y3 in itertools.product((-1, 1), repeat=6):
        vals = {"XXX": x1 * x2 * x3, "XYY": x1 * y2 * y3,
                "YXY": y1 * x2 * y3, "YYX": y1 * y2 * x3}
        k = sum(vals[key] == signs[key] for key in signs)
        best = max(best, k)
        n_all += k == 4
    return n_all, best


def pairwise_concurrences(state, n: int = 3) -> dict:
    rho = qm.as_dm(state)
    return {pair: qm.concurrence(reduced_n(rho, pair, n))
            for pair in itertools.combinations(range(n), 2)}


def component_graph(state, template: G.LocalityGraph, names=NAMES) -> G.LocalityGraph:
    """The rule of this row: the locality graph is the union of complete
    graphs on the entangled components of the state (weight 1 inside a
    component, no edge between components)."""
    g = G.LocalityGraph(dict(template.particles), {})
    for comp in entangled_components(state, len(names)):
        for i, j in itertools.combinations(comp, 2):
            g.link(names[i], names[j], 1.0)
    return g


def pairwise_weight_graph(state, template: G.LocalityGraph, names=NAMES) -> G.LocalityGraph:
    """The R05 rule applied blindly: w_ij = concurrence of the reduced pair."""
    g = G.LocalityGraph(dict(template.particles), {})
    for (i, j), c in pairwise_concurrences(state, len(names)).items():
        if c > 0:
            g.link(names[i], names[j], c)
    return g


def edges(g: G.LocalityGraph):
    return sorted(tuple(sorted(e)) for e, w in g.w.items() if w > 0)


# ------------------------------------------------------------------ claim 3
def measure_particle_1(n_dir, outcome: int = +1):
    """Post-measurement state after particle 1 is projected along n_dir,
    the 2-3 pair concurrence, and the entangled components."""
    rho, prob = project_particle(r11.ghz(), n_dir, outcome, 0, 3)
    c23 = qm.concurrence(reduced_n(rho, (1, 2), 3))
    return rho, prob, c23, entangled_components(rho, 3)


def pair_concurrence_vs_angle(thetas):
    """C_23 after particle 1 is projected along (sin t, 0, cos t)."""
    return np.array([measure_particle_1(qm.direction(t))[2] for t in thetas])


def postulate_ii_graph(template: G.LocalityGraph, measured: str) -> G.LocalityGraph:
    """Postulate (ii) alone: remove the measured particle's edges only."""
    g = G.LocalityGraph(dict(template.particles), dict(template.w))
    for j in g.partners(measured):
        g.unlink(measured, j)
    return g


# ------------------------------------------------------------------ claim 4
class StateHistory(G.History):
    """History whose measurement events replace the graph by the entangled
    components of the post-measurement state (M10 postulate ii read off the
    state), instead of only removing the measured particle's edges."""

    def __init__(self, graph: G.LocalityGraph, state, names=NAMES):
        super().__init__(graph)
        self.state = qm.as_dm(state)
        self.names = names
        self.state_graphs: dict[float, dict] = {}

    def measure(self, name: str, n_dir, outcome: int, y: float):
        which = self.names.index(name)
        self.state, _ = project_particle(self.state, n_dir, outcome, which, len(self.names))
        g = component_graph(self.state, self.graph, self.names)
        self.events.append(G.Event(y, "state", (name,)))
        self.state_graphs[y] = dict(g.w)
        return self

    def graph_at(self, y) -> G.LocalityGraph:
        g = super().graph_at(y)
        for e in sorted(self.events, key=lambda e: e.y):
            if e.y <= y and e.kind == "state":
                g.w = dict(self.state_graphs[e.y])
        return g


def measurement_kinematics(n_dir, y_m: float = 1.0, y_grid=None, eps: float = 1e-6):
    """Separations before/after a measurement of particle 1 at lab time
    y_m, in each frame; plus the position jump across the event."""
    y_grid = np.array([0.5, y_m, 2.0]) if y_grid is None else np.asarray(y_grid, dtype=float)
    g = triple()
    h = StateHistory(g, r11.ghz()).measure("P1", n_dir, +1, y_m)
    frames = list(g.particles) + [G.LAB]
    seps = {k: {pair: h.separation_in_frame(*pair, k, y_grid)
                for pair in (("P1", "P2"), ("P2", "P3"), ("P1", "P3"))} for k in frames}
    jump = max(abs(h.position_in_frame(j, k, [y_m + eps])[0] - h.position_in_frame(j, k, [y_m - eps])[0])
               for k in frames for j in g.particles)
    return {"graph_after": edges(h.graph_at(y_m)), "graph_before": edges(h.graph_at(y_m - eps)),
            "separations": seps, "jump": float(jump), "y": y_grid}


# ------------------------------------------------------------------ claim 5
def mermin_with_visibility(V: float) -> float:
    rho = V * qm.dm(r11.ghz()) + (1 - V) * np.eye(8) / 8
    return float(np.real(np.trace(rho @ r11.mermin_operator())))


def visibility_for_mermin(M: float) -> float:
    return M / 4.0


def correlator_from_fraction(f: float) -> float:
    """A fraction f of events following the GHZ sign gives E = f - (1 - f)."""
    return 2 * f - 1


# ------------------------------------------------------------------ claim 6
def n_particle_check(n: int) -> dict:
    g = r11.mutually_local_group(n)
    names = list(g.particles)
    return {
        "all_local": all(g.is_local(i, j) for i, j in itertools.combinations(names, 2)),
        "degree": r11.interpolant_degree(g),
        "distinct_lab_velocities": len({p.v for p in g.particles.values()}) == n,
    }


if __name__ == "__main__":
    g = triple()
    print("claim 1 velocities:", {k: {j: round(v, 3) for j, v in row.items()} for k, row in velocity_table(g).items()})
    print("        degree", r11.interpolant_degree(g))
    print("claim 2 signs", ghz_signs(), "LHV (all four, best)", lhv_assignments_satisfying())
    print("        pair concurrences", pairwise_concurrences(r11.ghz()))
    print("        component graph", edges(component_graph(r11.ghz(), g)), "pairwise-weight graph", edges(pairwise_weight_graph(r11.ghz(), g)))
    for lab, nd in (("Z", qm.direction(0.0)), ("X", qm.direction(np.pi / 2))):
        rho, p, c23, comps = measure_particle_1(nd)
        print(f"claim 3 along {lab}: p={p:.2f} C23={c23:.3f} components={comps} graph={edges(component_graph(rho, g))} postulate-ii={edges(postulate_ii_graph(g, 'P1'))}")
    print("        C23(theta):", pair_concurrence_vs_angle(np.linspace(0, np.pi, 5)))
    for lab, nd in (("Z", qm.direction(0.0)), ("X", qm.direction(np.pi / 2))):
        k = measurement_kinematics(nd)
        print(f"claim 4 {lab}:", k["graph_before"], "->", k["graph_after"], "jump", k["jump"])
        print("        sep P2-P3 in frame P2:", k["separations"]["P2"][("P2", "P3")], "lab:", k["separations"][G.LAB][("P2", "P3")])
    print("claim 5", {V: mermin_with_visibility(V) for V in (0.5, 0.71, 1.0)}, "V(2.77) =", visibility_for_mermin(2.77), "E(0.85) =", correlator_from_fraction(0.85))
    print("claim 6", {n: n_particle_check(n) for n in (3, 6, 14)})
