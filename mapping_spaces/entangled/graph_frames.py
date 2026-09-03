"""Pass 2 — frame potentials for N particles from a *locality graph*.

Generalises `transformation.md` (two particles at z = -/+1 with quadratic
potentials) to any number of particles on a one-dimensional qD.

Rule (plan §10.1).  Particle i has height z_i (≠ 0; the lab is at z = 0) and
lab velocity v_i.  Its frame potential sigma_i(z) is pinned at every height:

    sigma_i(z_i) = 0                          (i at rest in its own frame)
    sigma_i(z_j) = (1 - w_ij) (v_j - v_i)     (j seen with its relative
                                               velocity, scaled down by the
                                               locality weight w_ij)
    sigma_i(0)   = -v_i                       (the lab moves at -v_i)
    sigma_L(z_j) = v_j,  sigma_L(0) = 0       (the lab frame)

and interpolated between the pinned heights by the Lagrange polynomial (the
lowest-degree choice, as in transformation.md).  The shear from frame k to
frame l at height z is beta_kl(z) = sigma_l(z) - sigma_k(z), so composition
T_lm T_kl = T_km is automatic, and the velocity of particle j in frame k is
sigma_k(z_j).  For N = 2, z = -/+1, v = -/+1/a and w = 1 the potentials are
exactly those of transformation.md (tested).

Time dependence.  The graph changes at events (entangling, projection,
Bell-state measurement).  Potentials are piecewise constant in lab time y,
and positions are obtained by integrating: x^(k)_j(y) = x^lab_j(y) +
int_0^y beta_{L k}(z_j, y') dy', which keeps every position continuous.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

LAB = "L"


def lagrange(xs, ys):
    """Return the polynomial interpolant through (xs, ys) as a callable."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(set(xs.tolist())) != len(xs):
        raise ValueError("pinned heights must be distinct")

    def p(z):
        z = np.asarray(z, dtype=float)
        out = np.zeros_like(z)
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            term = np.full_like(z, yi)
            for j, xj in enumerate(xs):
                if j != i:
                    term = term * (z - xj) / (xi - xj)
            out = out + term
        return out

    return p


@dataclass
class Particle:
    name: str
    z: float
    v: float  # lab velocity along x
    x0: float = 0.0  # lab position at y = 0


@dataclass
class LocalityGraph:
    """Particles plus a symmetric weight w_ij in [0, 1] (1 = local partners)."""

    particles: dict = field(default_factory=dict)
    w: dict = field(default_factory=dict)  # frozenset({i, j}) -> weight

    def add(self, p: Particle):
        if p.z == 0.0:
            raise ValueError("z = 0 is the lab's height")
        for q in self.particles.values():
            if q.z == p.z and q.v != p.v:
                raise ValueError(
                    f"{p.name} and {q.name} share a height but have different lab "
                    "velocities: the lab potential would not be single-valued")
        self.particles[p.name] = p
        return self

    def weight(self, i, j) -> float:
        if i == j:
            return 1.0
        return self.w.get(frozenset((i, j)), 0.0)

    def link(self, i, j, weight: float = 1.0):
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must lie in [0, 1]")
        self.w[frozenset((i, j))] = weight
        return self

    def unlink(self, i, j):
        self.w.pop(frozenset((i, j)), None)
        return self

    def partners(self, i):
        return sorted(j for j in self.particles if j != i and self.weight(i, j) > 0)

    def component(self, i):
        """Particles connected to i by edges of positive weight."""
        seen, stack = {i}, [i]
        while stack:
            k = stack.pop()
            for j in self.partners(k):
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return sorted(seen)

    # --------------------------------------------------------- potentials
    def _heights(self):
        zs = [0.0] + [p.z for p in self.particles.values()]
        return sorted(set(zs))

    def potential(self, k):
        """sigma_k(z) as a callable (k a particle name or LAB)."""
        zs, vals = [], []
        for z in self._heights():
            if z == 0.0:
                vals.append(0.0 if k == LAB else -self.particles[k].v)
            else:
                j = next(n for n, p in self.particles.items() if p.z == z)
                pj = self.particles[j]
                if k == LAB:
                    vals.append(pj.v)
                elif j == k:
                    vals.append(0.0)
                else:
                    vals.append((1.0 - self.weight(k, j)) * (pj.v - self.particles[k].v))
            zs.append(z)
        return lagrange(zs, vals)

    def beta(self, k, l, z):
        """Shear coefficient from frame k to frame l at height z."""
        return self.potential(l)(z) - self.potential(k)(z)

    def velocity_in_frame(self, j, k) -> float:
        """Velocity of particle j (or the lab) as seen in frame k (= sigma_k(z_j))."""
        z = 0.0 if j == LAB else self.particles[j].z
        return float(self.potential(k)(z))

    def is_local(self, i, j, tol=1e-12) -> bool:
        """Mutually at rest in each other's frames."""
        return abs(self.velocity_in_frame(j, i)) < tol and abs(self.velocity_in_frame(i, j)) < tol

    def locality_fraction(self, i, j) -> float:
        """1 - (velocity of j in i's frame)/(lab relative velocity)."""
        rel = self.particles[j].v - self.particles[i].v
        if rel == 0:
            return 1.0
        return 1.0 - self.velocity_in_frame(j, i) / rel


# ------------------------------------------------------------ dynamics
@dataclass
class Event:
    y: float
    kind: str  # "entangle", "project", "bell"
    who: tuple
    weight: float = 1.0


def lab_position(p: Particle, y):
    return p.x0 + p.v * np.asarray(y, dtype=float)


class History:
    """A locality graph that changes at events; frame positions integrate
    the piecewise-constant shears so that they are continuous in y."""

    def __init__(self, graph: LocalityGraph, tol: float = 1e-9):
        self.graph = graph
        self.events: list[Event] = []
        self.tol = tol

    def _colocated(self, i, j, y) -> bool:
        pi, pj = self.graph.particles[i], self.graph.particles[j]
        return abs(lab_position(pi, y) - lab_position(pj, y)) < self.tol

    def entangle(self, i, j, y, weight=1.0):
        """Postulate (i): an edge is created only where the two particles are
        at the same lab position (feedback 7)."""
        if not self._colocated(i, j, y):
            raise ValueError(f"{i} and {j} are not co-located in the lab at y={y}")
        self.events.append(Event(y, "entangle", (i, j), weight))
        return self

    def project(self, i, y):
        """Postulate (ii): projecting i removes all of its edges."""
        self.events.append(Event(y, "project", (i,)))
        return self

    def bell_measure(self, b, c, y, transfer: bool = True):
        """Postulate (iii): a joint Bell-state measurement on co-located b, c
        removes their edges and, if ``transfer``, links their former partners."""
        if not self._colocated(b, c, y):
            raise ValueError(f"{b} and {c} are not co-located in the lab at y={y}")
        self.events.append(Event(y, "bell", (b, c), 1.0 if transfer else 0.0))
        return self

    def graph_at(self, y) -> LocalityGraph:
        g = LocalityGraph(dict(self.graph.particles), dict(self.graph.w))
        for e in sorted(self.events, key=lambda e: e.y):
            if e.y > y:
                break
            if e.kind == "entangle":
                g.link(*e.who, e.weight)
            elif e.kind == "project":
                for j in g.partners(e.who[0]):
                    g.unlink(e.who[0], j)
            elif e.kind == "bell":
                b, c = e.who
                pb = [j for j in g.partners(b) if j != c]
                pc = [j for j in g.partners(c) if j != b]
                for j in g.partners(b):
                    g.unlink(b, j)
                for j in g.partners(c):
                    g.unlink(c, j)
                if e.weight > 0:
                    for a in pb:
                        for d in pc:
                            g.link(a, d, 1.0)
        return g

    def breakpoints(self):
        return sorted({0.0, *(e.y for e in self.events)})

    def position_in_frame(self, j, k, y_grid):
        """x^(k)_j(y) on a grid, integrating beta_{L k}(z_j, y) piecewise."""
        y_grid = np.asarray(y_grid, dtype=float)
        pj = self.graph.particles[j]
        out = np.empty_like(y_grid)
        bps = self.breakpoints()
        for n, y in enumerate(y_grid):
            shift, y_prev = 0.0, 0.0
            for bp in bps + [np.inf]:
                if bp <= y_prev:
                    continue
                seg_end = min(bp, y)
                g = self.graph_at(y_prev)
                shift += g.beta(LAB, k, pj.z) * (seg_end - y_prev)
                y_prev = seg_end
                if seg_end >= y:
                    break
            out[n] = lab_position(pj, y) + shift
        return out

    def separation_in_frame(self, i, j, k, y_grid):
        return self.position_in_frame(j, k, y_grid) - self.position_in_frame(i, k, y_grid)


# ------------------------------------------------------------- helpers
def pass1_pair(a: float) -> LocalityGraph:
    """The transformation.md configuration: A at z=-1 (v=-1/a), B at z=+1
    (v=+1/a), fully local."""
    g = LocalityGraph()
    g.add(Particle("A", -1.0, -1.0 / a)).add(Particle("B", 1.0, 1.0 / a))
    g.link("A", "B", 1.0)
    return g
