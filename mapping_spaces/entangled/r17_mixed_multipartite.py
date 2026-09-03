"""R17 — the locality graph for *mixed* multipartite states (pass 3, plan §12):
definition by PPT cuts, and its decay under noise.

R13 defined the graph for pure states (finest product decomposition).  For a
mixed N-qubit state rho this module defines

  * membership: particles i and j are in the same component iff EVERY
    bipartition (cut) separating i from j is NPT (negative partial
    transpose); the components are the connected components of that relation
    (for the states studied here the relation is already transitive);
  * weight: a two-qubit component carries w = concurrence of its reduced
    state (R05 convention); a component S with |S| >= 3 carries
    w_S = min_{i in S} 2 N(rho_S; {i} | S \\ {i}), the normalised negativity
    of the weakest *single-particle* cut of the component's reduced state
    (a single qubit's negativity is at most 1/2, so w_S in [0, 1]).

Caveats (plan §12.1 Q1).  For two qubits PPT <=> separable (Peres–Horodecki),
so the rule is exact there; for N >= 3 a state can be PPT across every cut
and still be entangled (bound entanglement), so the rule can under-report
locality.  It is a definition, not a theorem.

Claims (numbered as in rows/R17_mixed_multipartite.md):
 1. Tools: partial transpose is an involution, trace- and Hermiticity-
    preserving; on random two-qubit states PPT <=> C = 0 and 2N <= C; the
    mixed-state component rule reproduces R13's pure-state components (GHZ,
    Bell (x) |0>, products, Bell (x) Bell on four qubits, the post-measurement
    states of R13 claim 3).  Pure GHZ_N: N = 1/2 across every cut; pure W_3:
    N = sqrt2/3 across every 1|2 cut.
 2. Noisy GHZ_3, rho_p = p|GHZ><GHZ| + (1-p) I/8: N_{i|jk}(p) =
    max(0, p/2 - (1-p)/8) for all three cuts, PPT threshold p* = 1/5
    (bisection); pairwise concurrences 0 for every p; components {1}{2}{3}
    for p <= p*, one component above, weight (5p-1)/4; projector witness
    <W> = 1/2 - F turns negative at p = 3/7; window (1/5, 3/7]: one
    component, not certified genuinely multipartite; Mermin > 2 needs p > 1/2.
 3. Noisy W_3: pairwise C(1) = 2/3, vanishing at p_C = (6 sqrt5 - 3)/19 =
    0.548; cut negativity vanishing at p_N = 3/(3 + 8 sqrt2) = 0.210; window
    (p_N, p_C]: component rule gives one component, pairwise rule no edge;
    projector witness threshold 13/21 = 0.619 > p_C.
 4. Local dephasing (phase flip q = (1 - e^{-gamma t})/2 per qubit): GHZ_N
    cut negativity = e^{-N gamma t}/2 exactly, never PPT at finite t —
    decay constant tau_N = 1/(N gamma), shrinking with N; W_3: pairwise
    C = (2/3) e^{-2 gamma t}, cut N = (sqrt2/3) e^{-2 gamma t}, tau = 1/(2 gamma)
    for any N.  Frame kinematics: the partner's separation in a member's
    frame is (v_j - v_i)[y - (1 - e^{-N gamma y})/(N gamma)] — a lag of
    (v_j - v_i)/(N gamma) behind the lab track.  Contrast: local
    depolarising kills the GHZ component at a finite r*_N (bisection).
 5. Ledger L-D6b/L-D6a: white-noise reading p = (F - 2^{-N})/(1 - 2^{-N});
    PPT threshold 1/(1 + 2^{N-1}) (bisection N = 2..6, formula for N = 14);
    every measured N is one component; the GME witness needs F > 1/2.
 6. The gap between the PPT threshold (~ 2^{1-N}) and the witness threshold
    ((2^{N-1} - 1)/(2^N - 1) -> 1/2) grows as ~ 2^{N-2}: the model's locality
    outlives certifiable multipartite entanglement.
"""
from __future__ import annotations

import itertools
import numpy as np

from mapping_spaces.entangled import graph_frames as G, qm
from mapping_spaces.entangled import r11_locality_graph as r11
from mapping_spaces.entangled import r13_ghz as r13

NAMES = r13.NAMES


# ------------------------------------------------------------------ states
def ghz(n: int) -> np.ndarray:
    v = np.zeros(2 ** n, dtype=complex)
    v[0] = v[-1] = 1 / np.sqrt(2)
    return v


def w_state(n: int) -> np.ndarray:
    v = np.zeros(2 ** n, dtype=complex)
    for k in range(n):
        v[1 << k] = 1 / np.sqrt(n)
    return v


def white_noise(state, p: float) -> np.ndarray:
    """p rho + (1 - p) I/d."""
    rho = qm.as_dm(state)
    d = rho.shape[0]
    return p * rho + (1 - p) * np.eye(d) / d


def fidelity_with(psi, rho) -> float:
    psi = np.asarray(psi, dtype=complex)
    return float(np.real(psi.conj() @ qm.as_dm(rho) @ psi))


# ------------------------------------------------------- partial transpose
def partial_transpose(rho, subset, n: int) -> np.ndarray:
    """Transpose the qubits in ``subset`` (0-based) of an n-qubit matrix."""
    t = np.asarray(rho, dtype=complex).reshape((2,) * (2 * n))
    for q in subset:
        t = np.swapaxes(t, q, n + q)
    return t.reshape(2 ** n, 2 ** n)


def negativity(rho, subset, n: int) -> float:
    """N = (||rho^{T_S}||_1 - 1)/2 = sum of |negative eigenvalues|."""
    ev = np.linalg.eigvalsh(partial_transpose(rho, subset, n))
    return float(-ev[ev < 0].sum())


def max_negativity(k: int, n: int) -> float:
    """Largest negativity across a k|(n-k) cut of qubits: (d_min - 1)/2."""
    return (2 ** min(k, n - k) - 1) / 2


def normalised_negativity(rho, subset, n: int) -> float:
    return negativity(rho, subset, n) / max_negativity(len(subset), n)


def is_ppt(rho, subset, n: int, tol: float = 1e-9) -> bool:
    return bool(np.linalg.eigvalsh(partial_transpose(rho, subset, n)).min() >= -tol)


def cuts(n: int):
    """Every bipartition, represented by the side that contains qubit 0."""
    others = list(range(1, n))
    for size in range(0, n - 1):
        for extra in itertools.combinations(others, size):
            yield (0, *extra)


def separating_cuts(i: int, j: int, n: int):
    return [S for S in cuts(n) if (i in S) != (j in S)]


# ---------------------------------------------------------- component rule
def same_component(rho, i: int, j: int, n: int, tol: float = 1e-9) -> bool:
    """i ~ j iff no cut separating them is PPT."""
    return all(not is_ppt(rho, S, n, tol) for S in separating_cuts(i, j, n))


def same_component_naive(rho, i: int, j: int, n: int, tol: float = 1e-9) -> bool:
    """NOT adopted: i ~ j iff *some* separating cut is NPT.  Shown in claim 1
    to merge two independent Bell pairs (the cut {i, j} | rest is NPT)."""
    return any(not is_ppt(rho, S, n, tol) for S in separating_cuts(i, j, n))


def components(rho, n: int, tol: float = 1e-9, relation=None):
    """Connected components of the relation ``same_component``.  Returns
    (list of sorted tuples, whether the relation was already transitive)."""
    rho = qm.as_dm(rho)
    relation = relation or same_component
    adj = {i: set() for i in range(n)}
    for i, j in itertools.combinations(range(n), 2):
        if relation(rho, i, j, n, tol):
            adj[i].add(j)
            adj[j].add(i)
    seen, comps = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], {i}
        while stack:
            k = stack.pop()
            for m in adj[k]:
                if m not in comp:
                    comp.add(m)
                    stack.append(m)
        seen |= comp
        comps.append(tuple(sorted(comp)))
    transitive = all(adj[i] >= (adj[k] - {i}) for i in adj for k in adj[i])
    return comps, transitive


def component_weight(rho, comp, n: int) -> float:
    """Definition of this row: concurrence for a pair; for |S| >= 3 the
    normalised negativity of the weakest single-particle cut of rho_S."""
    comp = tuple(comp)
    if len(comp) < 2:
        return 0.0
    rho_S = r13.reduced_n(qm.as_dm(rho), comp, n)
    if len(comp) == 2:
        return qm.concurrence(rho_S)
    m = len(comp)
    return float(min(2 * negativity(rho_S, (k,), m) for k in range(m)))


def weakest_cut_all(rho, comp, n: int) -> float:
    """Alternative (not adopted): min over *all* cuts of rho_S of the
    negativity normalised by that cut's own maximum."""
    comp = tuple(comp)
    m = len(comp)
    rho_S = r13.reduced_n(qm.as_dm(rho), comp, n)
    return float(min(normalised_negativity(rho_S, S, m) for S in cuts(m)))


def locality_graph(rho, template: G.LocalityGraph, names=NAMES, tol: float = 1e-9) -> G.LocalityGraph:
    """M13: complete graph on each PPT-cut component, edge weight = component weight."""
    n = len(names)
    g = G.LocalityGraph(dict(template.particles), {})
    for comp in components(rho, n, tol)[0]:
        w = component_weight(rho, comp, n)
        for i, j in itertools.combinations(comp, 2):
            if w > 0:
                g.link(names[i], names[j], w)
    return g


def pairwise_graph(rho, template: G.LocalityGraph, names=NAMES) -> G.LocalityGraph:
    """The R05 rule applied blindly (w_ij = concurrence of the reduced pair)."""
    return r13.pairwise_weight_graph(rho, template, names)


edges = r13.edges


def edge_weights(g: G.LocalityGraph) -> dict:
    return {tuple(sorted(e)): w for e, w in g.w.items() if w > 0}


# ----------------------------------------------------------- claim-1 checks
def random_two_qubit_states(m: int, seed: int = 17):
    """m random mixed two-qubit states (Hilbert–Schmidt-like, rank 1..4) and
    m random pure ones."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(m):
        k = int(rng.integers(1, 5))
        A = rng.normal(size=(4, k)) + 1j * rng.normal(size=(4, k))
        rho = A @ A.conj().T
        out.append(rho / np.trace(rho).real)
    for _ in range(m):
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        out.append(qm.dm(v / np.linalg.norm(v)))
    return out


def two_qubit_checks(m: int = 150, seed: int = 17) -> dict:
    """PPT <=> C = 0 (Peres–Horodecki) and 2N <= C on random states."""
    agree, bound, n_ent = 0, 0, 0
    for rho in random_two_qubit_states(m, seed):
        C = qm.concurrence(rho)
        N2 = 2 * negativity(rho, (0,), 2)
        ppt = is_ppt(rho, (0,), 2)
        agree += (ppt == (C < 1e-7))
        bound += (N2 <= C + 1e-6)  # equality for pure states, to concurrence precision
        n_ent += C > 1e-7
    return {"states": 2 * m, "ppt_iff_separable": agree, "2N_le_C": bound, "entangled": n_ent}


def pure_state_cases():
    """(name, state, n, R13 components) for the cross-check of claim 1."""
    bell = qm.singlet()
    zero = qm.ket(1, 0)
    cases = [
        ("GHZ_3", ghz(3), 3), ("W_3", w_state(3), 3),
        ("Bell(x)|0>", np.kron(bell, zero), 3),
        ("|0>(x)Bell", np.kron(zero, bell), 3),
        ("product", np.kron(np.kron(zero, qm.ket(1, 1)), qm.ket(1, 1j)), 3),
        ("Bell(x)Bell", np.kron(bell, bell), 4),
        ("GHZ_4", ghz(4), 4),
    ]
    for lab, nd in (("GHZ after Z on P1", qm.direction(0.0)), ("GHZ after X on P1", qm.direction(np.pi / 2))):
        cases.append((lab, r13.measure_particle_1(nd)[0], 3))
    return [(lab, st, n, r13.entangled_components(st, n)) for lab, st, n in cases]


# ------------------------------------------------------------- thresholds
def bisect(f, lo: float, hi: float, tol: float = 1e-10) -> float:
    """Root of f on [lo, hi] with f(lo) <= 0 < f(hi) (or the reverse)."""
    flo, fhi = f(lo), f(hi)
    if (flo > 0) == (fhi > 0):
        raise ValueError("no sign change on the bracket")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (fhi > 0):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def cut_negativities(state, p: float, n: int) -> dict:
    rho = white_noise(state, p)
    return {(k,): negativity(rho, (k,), n) for k in range(n)}


def ghz_negativity_closed(p: float, n: int) -> float:
    return max(0.0, p / 2 - (1 - p) / 2 ** n)


def ppt_threshold(state, n: int, cut=(0,)) -> float:
    """Smallest p above which p|psi><psi| + (1-p) I/d is NPT across ``cut``."""
    return bisect(lambda p: negativity(white_noise(state, p), cut, n) - 1e-14, 0.0, 1.0)


def ghz_ppt_threshold_formula(n: int) -> float:
    return 1.0 / (1 + 2 ** (n - 1))


def pairwise_concurrences(state, p: float, n: int = 3) -> dict:
    return r13.pairwise_concurrences(white_noise(state, p), n)


def ghz_component_weight_closed(p: float) -> float:
    return max(0.0, (5 * p - 1) / 4)


def biseparable_overlap(psi, n: int) -> float:
    """alpha = max over biseparable states of |<psi|phi>|^2 = largest
    eigenvalue of a reduced state, maximised over cuts (projector-witness
    constant, computed rather than quoted)."""
    rho = qm.dm(psi)
    return float(max(np.linalg.eigvalsh(r13.reduced_n(rho, S, n)).max() for S in cuts(n)))


def projector_witness(psi, rho, n: int) -> float:
    """<W> = alpha - F; negative certifies genuine multipartite entanglement."""
    return biseparable_overlap(psi, n) - fidelity_with(psi, rho)


def witness_threshold(psi, n: int) -> float:
    return bisect(lambda p: -projector_witness(psi, white_noise(psi, p), n), 0.0, 1.0)


def ghz_witness_threshold_formula(n: int) -> float:
    return (2 ** (n - 1) - 1) / (2 ** n - 1)


def mermin_threshold() -> float:
    """R13 claim 5: Mermin value 4p exceeds the local bound 2 for p > 1/2."""
    return bisect(lambda p: r13.mermin_with_visibility(p) - 2.0, 0.0, 1.0)


def w_pair_threshold_formula() -> float:
    return (6 * np.sqrt(5) - 3) / 19


def w_cut_threshold_formula() -> float:
    return 3 / (3 + 8 * np.sqrt(2))


def w_pair_threshold() -> float:
    return bisect(lambda p: pairwise_concurrences(w_state(3), p)[(0, 1)] - 1e-14, 0.0, 1.0)


def w_witness_threshold_formula() -> float:
    return 13 / 21


# ---------------------------------------------------------------- dynamics
def _apply_local(rho, n: int, kraus):
    for q in range(n):
        out = np.zeros_like(rho)
        for K in kraus:
            ops = [qm.I2] * n
            ops[q] = K
            Kq = qm.kron(*ops)
            out = out + Kq @ rho @ Kq.conj().T
        rho = out
    return rho


def dephase(rho, q: float, n: int) -> np.ndarray:
    """Phase-flip channel with probability q on every qubit."""
    rho = qm.as_dm(rho)
    return _apply_local(rho, n, [np.sqrt(1 - q) * qm.I2, np.sqrt(q) * qm.SZ])


def depolarise(rho, r: float, n: int) -> np.ndarray:
    """Each qubit replaced by I/2 with probability r."""
    rho = qm.as_dm(rho)
    ks = [np.sqrt(1 - 3 * r / 4) * qm.I2] + [np.sqrt(r / 4) * P for P in (qm.SX, qm.SY, qm.SZ)]
    return _apply_local(rho, n, ks)


def q_of_t(gamma: float, t) -> np.ndarray:
    return (1 - np.exp(-gamma * np.asarray(t, dtype=float))) / 2


def dephasing_trace(state, n: int, gamma: float, ts) -> dict:
    """Cut negativity (qubit 0 | rest), component weight and pairwise
    concurrence (0,1) of a dephased state on a time grid."""
    ts = np.asarray(ts, dtype=float)
    neg, wgt, conc, ppt = [], [], [], []
    for t in ts:
        rho = dephase(state, float(q_of_t(gamma, t)), n)
        neg.append(negativity(rho, (0,), n))
        ppt.append(is_ppt(rho, (0,), n))
        comps, _ = components(rho, n)
        wgt.append(component_weight(rho, comps[0], n) if len(comps[0]) == n else 0.0)
        conc.append(qm.concurrence(r13.reduced_n(rho, (0, 1), n)))
    return {"t": ts, "negativity": np.array(neg), "weight": np.array(wgt),
            "concurrence": np.array(conc), "ppt": np.array(ppt)}


def ghz_dephasing_closed(n: int, gamma: float, ts) -> np.ndarray:
    return 0.5 * np.exp(-n * gamma * np.asarray(ts, dtype=float))


def w3_dephasing_closed(gamma: float, ts) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(ts, dtype=float)
    return np.sqrt(2) / 3 * np.exp(-2 * gamma * ts), 2 / 3 * np.exp(-2 * gamma * ts)


def lifetime(n: int, gamma: float) -> float:
    """1/e time of the GHZ_N component weight under local dephasing."""
    return 1.0 / (n * gamma)


def time_to_weight(n: int, gamma: float, eps: float) -> float:
    return np.log(1 / eps) / (n * gamma)


def depolarising_death(state, n: int, cut=(0,)) -> float:
    """Per-qubit depolarising probability r* at which the cut turns PPT."""
    return bisect(lambda r: 1e-14 - negativity(depolarise(state, r, n), cut, n), 0.0, 1.0)


def component_death_depolarising(state, n: int) -> tuple[float, tuple]:
    """Smallest r* over all cuts (the component dissolves when the first
    separating cut turns PPT)."""
    best = min(((depolarising_death(state, n, S), S) for S in cuts(n)), key=lambda t: t[0])
    return best


def depolarising_death_time(r_star: float, Gamma: float) -> float:
    """r = 1 - e^{-Gamma t}  =>  t* = -ln(1 - r*)/Gamma."""
    return -np.log(1 - r_star) / Gamma


def separation_under_decay(gamma: float, y_grid, i: str = "P1", j: str = "P2", n: int = 3,
                           template: G.LocalityGraph | None = None) -> np.ndarray:
    """Separation of j from i in i's frame when every edge of the triple
    carries w(y) = e^{-n gamma y}: integral of the velocity of j in frame i
    (graph_frames potentials), trapezoid rule on the grid."""
    g = template or r13.triple()
    y_grid = np.asarray(y_grid, dtype=float)
    vel = np.empty_like(y_grid)
    for k, y in enumerate(y_grid):
        w = float(np.exp(-n * gamma * y))
        for a, b in itertools.combinations(list(g.particles), 2):
            g.link(a, b, w)
        vel[k] = g.velocity_in_frame(j, i)
    dy = np.diff(y_grid)
    sep = np.concatenate([[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1]) * dy)])
    return sep


def separation_closed(gamma: float, y_grid, vi: float, vj: float, n: int = 3) -> np.ndarray:
    y = np.asarray(y_grid, dtype=float)
    return (vj - vi) * (y - (1 - np.exp(-n * gamma * y)) / (n * gamma))


def frame_lag(gamma: float, vi: float, vj: float, n: int = 3) -> float:
    return (vj - vi) / (n * gamma)


# ------------------------------------------------------------------ ledger
# L-D6b (Monz et al. 2011, VERIFIED): fidelity (value, error) for N = 2, 3, 4, 14.
# The ledger entry lists only these four; N = 5..13 are elided there ("...").
LEDGER_D6B = {2: (0.986, 0.002), 3: (0.970, 0.003), 4: (0.957, 0.003), 14: (0.508, 0.009)}
# L-D6a (Leibfried et al. 2005, PARTIAL): fidelity lower bounds for N = 4, 5, 6.
LEDGER_D6A = {4: (0.76, 0.01), 5: (0.60, 0.02), 6: (0.509, 0.004)}


def p_from_fidelity(F: float, n: int) -> float:
    """White-noise model: F = p + (1 - p)/2^N."""
    return (F - 2.0 ** -n) / (1 - 2.0 ** -n)


def ledger_table(bisect_up_to: int = 6) -> list:
    rows = []
    for entry, table in (("L-D6b", LEDGER_D6B), ("L-D6a", LEDGER_D6A)):
        for n, (F, dF) in table.items():
            p = p_from_fidelity(F, n)
            p_ppt = ppt_threshold(ghz(n), n) if n <= bisect_up_to else None
            rows.append({
                "entry": entry, "N": n, "F": F, "dF": dF, "p": p,
                "p_ppt_bisection": p_ppt, "p_ppt_formula": ghz_ppt_threshold_formula(n),
                "p_gme_formula": ghz_witness_threshold_formula(n),
                "one_component": p > ghz_ppt_threshold_formula(n),
                "gme_certified": F > 0.5,
                "gme_sigma": (F - 0.5) / dF,
            })
    return rows


def threshold_gap(n: int) -> float:
    return ghz_witness_threshold_formula(n) / ghz_ppt_threshold_formula(n)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    g = r13.triple()
    print("claim 1:", two_qubit_checks())
    for lab, st, n, comps in pure_state_cases():
        print(f"         {lab}: R13 {comps} mixed-rule {components(st, n)} naive {components(st, n, relation=same_component_naive)[0]}")
    print("claim 1: GHZ_3 cut negativities", cut_negativities(ghz(3), 1.0, 3), "W_3", cut_negativities(w_state(3), 1.0, 3))
    print("         GHZ_4 normalised 1|3", normalised_negativity(qm.dm(ghz(4)), (0,), 4), "2|2", normalised_negativity(qm.dm(ghz(4)), (0, 1), 4))
    print("claim 2: p* =", ppt_threshold(ghz(3), 3), "formula", ghz_ppt_threshold_formula(3), "witness", witness_threshold(ghz(3), 3), "Mermin", mermin_threshold())
    for p in (0.1, 0.2, 0.3, 3 / 7, 0.5, 1.0):
        rho = white_noise(ghz(3), p)
        print(f"   p={p:.3f} neg={cut_negativities(ghz(3), p, 3)[(0,)]:.4f} comps={components(rho, 3)[0]} w={edge_weights(locality_graph(rho, g))} pairs={pairwise_concurrences(ghz(3), p)} <W>={projector_witness(ghz(3), rho, 3):+.4f}")
    print("claim 3: W p_N =", ppt_threshold(w_state(3), 3), w_cut_threshold_formula(), "p_C =", w_pair_threshold(), w_pair_threshold_formula(), "witness", witness_threshold(w_state(3), 3), w_witness_threshold_formula())
    ts = np.linspace(0, 2, 5)
    for n in (2, 3, 4):
        tr = dephasing_trace(ghz(n), n, 1.0, ts)
        print(f"claim 4: GHZ_{n} dephasing neg", tr["negativity"], "closed", ghz_dephasing_closed(n, 1.0, ts), "ppt", tr["ppt"], "tau", lifetime(n, 1.0), "r*", component_death_depolarising(ghz(n), n))
    tw = dephasing_trace(w_state(3), 3, 1.0, ts)
    print("         W_3 neg", tw["negativity"], "C", tw["concurrence"], "closed", w3_dephasing_closed(1.0, ts))
    y = np.linspace(0, 3, 3001)
    print("         lag", separation_under_decay(1.0, y)[-1], separation_closed(1.0, y, -1.0, 1.0)[-1], frame_lag(1.0, -1.0, 1.0))
    for row in ledger_table():
        print("claim 5:", row)
    print("claim 6: gap", {n: threshold_gap(n) for n in (2, 3, 4, 6, 14)})
