"""R11 — locality-graph potentials: N particles on a one-dimensional qD.

Claims (numbered as in rows/R11_locality_graph.md):
 1. For N = 2 at z = -/+1 the graph rule reproduces transformation.md's
    potentials exactly (all three: A, B, lab).
 2. Every particle sees every non-partner at exactly its lab-relative
    velocity, the lab at -v_i, and each partner at rest: the pinned values
    are reproduced by the interpolant (no distortion at pinned heights).
 3. Three particles created together can be mutually local at three
    distinct heights of a 1-D qD (no second qD dimension needed); the same
    holds for N = 4, 5.  The interpolant degree grows as N (pinned points
    N + 1), which is the cost of encoding the graph in one dimension.
 4. Same height forces the same lab velocity (the lab potential must be
    single-valued); non-co-moving particles therefore need distinct heights,
    and two particles at the same height are automatically mutually at rest.
 5. Locality fraction is linear in the weight: l(w) = w (settles R05 in
    favour of the pinned-lab convention).
 6. Statistics are QM's: the GHZ state gives Mermin <M> = 4 (LHV bound 2);
    the graph adds no prediction to the spin sector.
"""
from __future__ import annotations

import itertools
import numpy as np

from mapping_spaces import transformation as T
from mapping_spaces.entangled import graph_frames as G, qm


def claim1_reduction(a: float = 2.0, z=None):
    z = np.linspace(-2, 2, 21) if z is None else np.asarray(z)
    g = G.pass1_pair(a)
    return {
        "A": (g.potential("A")(z), T.shear_potential(1, a)(z)),
        "B": (g.potential("B")(z), T.shear_potential(2, a)(z)),
        "L": (g.potential(G.LAB)(z), T.shear_potential(3, a)(z)),
    }


def example_graph():
    """Four particles: A~B (a pair), C~D (a pair), E unrelated."""
    g = G.LocalityGraph()
    g.add(G.Particle("A", -1.0, -1.0)).add(G.Particle("B", 1.0, 1.0))
    g.add(G.Particle("C", -2.0, -0.4)).add(G.Particle("D", 2.0, 0.7))
    g.add(G.Particle("E", 0.5, 0.25))
    g.link("A", "B").link("C", "D")
    return g


def claim2_velocity_table(g=None):
    """Matrix V[i][j] = velocity of j in frame i (j may be the lab)."""
    g = g or example_graph()
    names = list(g.particles) + [G.LAB]
    V = {}
    for i in g.particles:
        V[i] = {j: g.velocity_in_frame(j, i) for j in names}
    return V


def expected_velocity(g, i, j):
    if j == G.LAB:
        return -g.particles[i].v
    return (1 - g.weight(i, j)) * (g.particles[j].v - g.particles[i].v)


def mutually_local_group(n: int):
    """n particles at distinct heights 1..n with distinct velocities, all linked."""
    g = G.LocalityGraph()
    for k in range(n):
        g.add(G.Particle(f"P{k}", float(k + 1), 0.3 * (k - (n - 1) / 2)))
    for i, j in itertools.combinations(range(n), 2):
        g.link(f"P{i}", f"P{j}")
    return g


def claim3_all_local(n: int) -> bool:
    g = mutually_local_group(n)
    return all(g.is_local(f"P{i}", f"P{j}") for i, j in itertools.combinations(range(n), 2))


def interpolant_degree(g: G.LocalityGraph) -> int:
    return len({0.0, *(p.z for p in g.particles.values())}) - 1


def claim5_locality_fraction(ws):
    return [G.pass1_pair(1.0).link("A", "B", w).locality_fraction("A", "B") for w in ws]


def ghz() -> np.ndarray:
    v = np.zeros(8, dtype=complex)
    v[0] = v[7] = 1 / np.sqrt(2)
    return v


def mermin_operator() -> np.ndarray:
    X, Y = qm.SX, qm.SY
    return qm.kron(X, X, X) - qm.kron(X, Y, Y) - qm.kron(Y, X, Y) - qm.kron(Y, Y, X)


def claim6_mermin() -> float:
    psi = ghz()
    return float(np.real(psi.conj() @ mermin_operator() @ psi))


def mermin_lhv_max() -> int:
    """Max of <M> over deterministic local assignments x_i, y_i in {+-1}."""
    best = -8
    for vals in itertools.product((-1, 1), repeat=6):
        x1, y1, x2, y2, x3, y3 = vals
        best = max(best, x1 * x2 * x3 - x1 * y2 * y3 - y1 * x2 * y3 - y1 * y2 * x3)
    return best


if __name__ == "__main__":
    for k, (got, want) in claim1_reduction().items():
        print(k, "max|diff|", np.abs(got - want).max())
    g = example_graph()
    for i, row in claim2_velocity_table(g).items():
        print(i, {j: round(v, 6) for j, v in row.items()})
    for n in (3, 4, 5):
        print(n, "mutually local:", claim3_all_local(n), "degree", interpolant_degree(mutually_local_group(n)))
    print("locality fraction", claim5_locality_fraction([0, 0.25, 0.5, 1]))
    print("Mermin GHZ", claim6_mermin(), "LHV max", mermin_lhv_max())
