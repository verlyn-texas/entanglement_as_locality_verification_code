"""R21 — a continuous locality-weight rule (paper-1 revision, handoff B3/D6).

The evaluation exhibited (checks/check_states.py part E) a discontinuity of the
min-cut *component* weight: for sqrt(1-eps^2) (singlet_12 (x) |0>_3) + eps|111>
the three-particle component weight -> 0 as eps -> 0, yet at eps = 0 the
component splits and the pair rule gives w_12 = C = 1.  The repair adopted here
replaces the per-component weight by a per-edge weight,

    w_ij(rho) = 2 * min over bipartite cuts (P, Pbar) separating i from j
                of  N_cut(rho),

with N_cut the negativity across the cut.  Properties (each tested):

* continuity: N across any fixed cut is continuous in rho, and a minimum of
  finitely many continuous functions is continuous, so every w_ij — and with it
  every pinned potential value (1 - w_ij)(v_j - v_i) — is continuous in the
  state.  "Nothing jumps" becomes a theorem instead of a broken slogan.
* the D6 counterexample is repaired: w_12 -> 1 smoothly as eps -> 0 (the cuts
  separating 1 from 2 keep negativity ~ 1/2), while w_13, w_23 -> 0 smoothly.
* a lone pair gets w = 2 N(rho), which equals the concurrence on the two
  families the paper quantifies (pure two-qubit states and Werner states) and
  is <= C in general.
* GHZ_N keeps every edge at 1 (all separating cuts have negativity 1/2), and
  measuring one particle of GHZ_3 along Z still removes the far edge.
* noisy-GHZ thresholds are unchanged: the minimal separating cut is the
  single-particle cut, so w > 0 exactly above the old component threshold.
* the weights inside a >= 3-particle component are pairwise by construction —
  resolving the uniform-vs-pairwise ambiguity the referees flagged.

Edges with w_ij > 0 still define the components (postulate P2); the rule is
still blind to bound entanglement (all-PPT states get an empty graph).

Round-2 revision (R4/B-M2): beyond qubits the bare rule w = 2 N_min exceeds 1
(a maximally entangled spin-1 pair has every separating-cut negativity 1, so
w = 2, and P1 would pin the partner at velocity fraction 1 - w = -1).  The
normalised rule adopted is

    w_ij(rho) = 2 * N_min / (d_min - 1),   d_min = min(d_i, d_j),

with d_i the local (spin) dimension of particle i.  Boundedness: both
single-particle cuts {i} and {j} separate i from j, and the negativity of a
cut is at most (min cut dimension - 1)/2, so N_min <= (d_min - 1)/2 and
w_ij <= 1.  For qubits d_min - 1 = 1 and the rule is unchanged — every
qubit-based result above (continuity, the spectator family, the l = C
coincidences, the noisy-GHZ thresholds) survives verbatim.  Continuity holds
with the same proof: the divisor is a constant per edge.
"""
from __future__ import annotations

import numpy as np

from . import r17_mixed_multipartite as r17


def edge_weight(rho: np.ndarray, i: int, j: int, n: int) -> float:
    """w_ij for n qubits: 2 * min_{cuts separating i, j} negativity(rho, cut)
    (the d_min - 1 divisor of the normalised rule is 1 for qubits)."""
    return 2.0 * min(r17.negativity(rho, cut, n) for cut in r17.separating_cuts(i, j, n))


# ------------------------------------------------- general local dimensions
def partial_transpose_dims(rho: np.ndarray, subset, dims) -> np.ndarray:
    """Partial transpose over ``subset`` for particles of local dimensions
    ``dims`` (a tuple; generalises r17.partial_transpose beyond qubits)."""
    n = len(dims)
    t = np.asarray(rho, dtype=complex).reshape(*dims, *dims)
    for q in subset:
        t = np.swapaxes(t, q, n + q)
    d = int(np.prod(dims))
    return t.reshape(d, d)


def negativity_dims(rho: np.ndarray, subset, dims) -> float:
    ev = np.linalg.eigvalsh(partial_transpose_dims(rho, subset, dims))
    return float(-ev[ev < 0].sum())


def edge_weight_dims(rho: np.ndarray, i: int, j: int, dims) -> float:
    """The normalised P2 rule: w_ij = 2 N_min / (d_min - 1) with
    d_min = min(dims[i], dims[j]).  Always in [0, 1]; equals edge_weight
    when every particle is a qubit."""
    n = len(dims)
    n_min = min(negativity_dims(rho, cut, dims) for cut in r17.separating_cuts(i, j, n))
    return 2.0 * n_min / (min(dims[i], dims[j]) - 1)


def max_entangled_pair(d: int) -> np.ndarray:
    """|Phi_d> = sum_k |kk>/sqrt(d) as a density matrix (spin-1: d = 3)."""
    psi = np.zeros(d * d, complex)
    psi[:: d + 1] = 1 / np.sqrt(d)
    return np.outer(psi, psi.conj())


def unnormalised_pair_weight(d: int) -> float:
    """The refuted pre-revision value 2 N for a maximally entangled qudit
    pair: d - 1 (= 2 for spin 1), the R4 counterexample."""
    return 2.0 * negativity_dims(max_entangled_pair(d), (0,), (d, d))


def all_edge_weights(rho: np.ndarray, n: int) -> dict:
    return {(i, j): edge_weight(rho, i, j, n) for i in range(n) for j in range(i + 1, n)}


def spectator_state(eps: float) -> np.ndarray:
    """sqrt(1-eps^2) (singlet_12 (x) |0>_3) + eps |111> — the D6 counterexample."""
    psi = np.zeros(8, complex)
    # singlet (|01> - |10>)/sqrt2 on qubits 1,2 with |0> on qubit 3
    psi[0b010] = np.sqrt(1 - eps**2) / np.sqrt(2)
    psi[0b100] = -np.sqrt(1 - eps**2) / np.sqrt(2)
    psi[0b111] = eps
    return np.outer(psi, psi.conj())


def spectator_weights(eps: float) -> dict:
    return all_edge_weights(spectator_state(eps), 3)


def old_component_weight(eps: float) -> float:
    """The pre-revision min-cut component weight of the eps > 0 state
    (2 x weakest single-particle-cut negativity), for contrast."""
    rho = spectator_state(eps)
    return 2.0 * min(r17.negativity(rho, (k,), 3) for k in range(3))


def pair_weight_two_qubits(rho4: np.ndarray) -> float:
    """The rule specialised to a lone pair: 2 N(rho)."""
    return edge_weight(rho4, 0, 1, 2)
