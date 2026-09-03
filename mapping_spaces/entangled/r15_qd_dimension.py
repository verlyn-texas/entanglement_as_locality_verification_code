"""R15 — does the *dimension* of qD carry any content?

The locality-graph rule (R11) pins the frame potential of every particle at
the heights of all particles and the lab; the interpolant between pinned
points is a free choice.  A 2-D qD replaces heights z_i by points (z_i, w_i)
and the univariate interpolant by a bivariate one.

Claims (numbered as in rows/R15_qd_dimension.md):
 1. In 1-D, N particles + lab pin N+1 points, so the minimal polynomial
    degree is N.  In 2-D a total-degree-d polynomial has (d+1)(d+2)/2
    coefficients, so N+1 points in general position need only
    d = ceil((sqrt(8N+9)-3)/2) ~ sqrt(2N): 2-D lowers the degree.
 2. Every pinned value (velocity of each particle/lab in each frame) is the
    same in 1-D and 2-D, so every locality relation, every lab track and
    every frame position of a particle is identical: the embedding is
    unobservable.
 3. Only relationships placed at NON-pinned points (transformation.md §5.2)
    differ between embeddings, and nothing physical sits there.
 4. The 1-D obstruction "same height => same lab velocity" (R11 claim 4)
    disappears in 2-D (points may share z with different w), but it was
    never an empirical constraint — heights are labels.
 5. Hence the qD dimension is not a parameter of the model; the requirement's
    "higher dimensions of qD for larger entangled sets" has no content beyond
    interpolant convenience.
"""
from __future__ import annotations

import itertools
import math
import numpy as np

from mapping_spaces.entangled import graph_frames as G


def degree_1d(n_particles: int) -> int:
    return n_particles  # N+1 pinned points


def degree_2d(n_particles: int) -> int:
    """Smallest total degree d with (d+1)(d+2)/2 >= N+1."""
    d = 0
    while (d + 1) * (d + 2) // 2 < n_particles + 1:
        d += 1
    return d


def _monomials(d: int):
    return [(a, b) for a in range(d + 1) for b in range(d + 1 - a)]


def bivariate_interpolant(points, values, d: int):
    """Least-squares polynomial of total degree d through (z, w) -> value;
    exact when the points are in general position and count <= #monomials."""
    mons = _monomials(d)
    A = np.array([[p[0] ** a * p[1] ** b for (a, b) in mons] for p in points], dtype=float)
    coef, *_ = np.linalg.lstsq(A, np.asarray(values, dtype=float), rcond=None)

    def f(z, w):
        return sum(c * z ** a * w ** b for c, (a, b) in zip(coef, mons))

    return f, np.abs(A @ coef - values).max()


def embed_2d(g: G.LocalityGraph, rng_seed: int = 3):
    """Assign each particle a second coordinate w (random, distinct) and
    return the 2-D potentials {frame: f(z, w)} plus the max fit residual."""
    rng = np.random.default_rng(rng_seed)
    pts = {G.LAB: (0.0, 0.0)}
    for name, p in g.particles.items():
        pts[name] = (p.z, float(rng.uniform(-1, 1)))
    d = degree_2d(len(g.particles))
    pots, worst = {}, 0.0
    for k in list(g.particles) + [G.LAB]:
        sigma = g.potential(k)  # 1-D potential: pinned values are what we copy
        vals = [float(sigma(0.0)) if j == G.LAB else float(sigma(g.particles[j].z))
                for j in pts]
        f, res = bivariate_interpolant(list(pts.values()), vals, d)
        pots[k] = f
        worst = max(worst, res)
    return pts, pots, worst


def pinned_table(g: G.LocalityGraph):
    """velocity of j in frame k, from the 1-D potentials."""
    names = list(g.particles) + [G.LAB]
    return {k: {j: g.velocity_in_frame(j, k) for j in names} for k in g.particles}


def pinned_table_2d(g: G.LocalityGraph, pts, pots):
    names = list(g.particles) + [G.LAB]
    return {k: {j: float(pots[k](*pts[j])) for j in names} for k in g.particles}


def off_pinned_difference(g: G.LocalityGraph, pts, pots, k, z: float, w: float) -> float:
    """|sigma_k^{2D}(z, w) - sigma_k^{1D}(z)| at a point where nothing sits."""
    return abs(float(pots[k](z, w)) - float(g.potential(k)(z)))


def example():
    g = G.LocalityGraph()
    g.add(G.Particle("A", -1.0, -1.0)).add(G.Particle("B", 1.0, 1.0))
    g.add(G.Particle("C", -2.0, -0.4)).add(G.Particle("D", 2.0, 0.7)).add(G.Particle("E", 0.5, 0.25))
    g.link("A", "B").link("C", "D")
    return g


if __name__ == "__main__":
    for n in (2, 3, 5, 10, 20, 50):
        print(n, "degree 1-D", degree_1d(n), "2-D", degree_2d(n))
    g = example()
    pts, pots, worst = embed_2d(g)
    print("fit residual", worst)
    t1, t2 = pinned_table(g), pinned_table_2d(g, pts, pots)
    print("max pinned difference", max(abs(t1[k][j] - t2[k][j]) for k in t1 for j in t1[k]))
    print("off-pinned difference at (0.3, 0.3) in A:", off_pinned_difference(g, pts, pots, "A", 0.3, 0.3))
