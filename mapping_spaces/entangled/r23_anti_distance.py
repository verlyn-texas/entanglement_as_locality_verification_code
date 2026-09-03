"""R23 — the anti-"distance" programme: three results (paper-1 revision,
handoff Tier C3-C5, executed per the user's Q1 answer).

C3 — a contact-point-supported generator of the update (Proposition 5).
    The joint projection of a pair at the first measurement is effected by the
    operator M_s (x) 1 supported on the measured particle alone, so a
    stochastic jump process triggered at the measurement event — which, in the
    pair's chart, lies at the one point both particles occupy — generates the
    update locally *in the chart*, and its pushforward to the laboratory chart
    is exactly the projection rule.  Locality of *implementation* is earned.
    Its limit: by Proposition 2 (r02), any outcome-producing rule at the
    contact point that reproduces the singlet box must use the quantum
    conditionals; the jump law is necessarily quantum-tuned, so locality of
    *explanation* is not earned, provably.

C4 — order-independence narrows the polytope (Proposition 6).
    A contact rule with A measured first realises exactly the boxes whose
    A-marginal ignores B's setting (one-way no-signalling; Proposition 2).
    In the mechanism the measurement order varies run to run (it is set by
    times/worldlines, not by the physics of the pair), and no timing
    dependence of the box is observed or predicted; demanding that the SAME
    box be realisable with either order is therefore forced.  A box is
    realisable in both directions iff it is fully no-signalling (chain rule in
    both factorisation orders).  So the geometry-native axiom "the box does
    not remember who was measured first" cuts the one-way-no-signalling
    polytope exactly down to the no-signalling polytope — a genuine narrowing,
    which still stops short of the quantum set (the PR box survives).

C5 — no state-independent QRF reproduces the chart (Proposition 7).
    The chart velocity of a partner is (1 - w(rho)) * (relative velocity) with
    w(rho) = 2 N(rho), a NON-AFFINE functional of the state (for the Werner
    family, w(p) = max(0, (3p-1)/2) has a kink at p = 1/3).  Any fixed unitary
    V — a quantum-reference-frame transformation in the sense of Giacomini,
    Castro-Ruiz and Brukner — or indeed any fixed linear channel induces
    AFFINE functionals of the state: <O>_{V rho V^+} is linear in rho.  Hence
    no state-independent QRF transformation has the label shear as a classical
    limit; only a state-dependent construction can.
"""
from __future__ import annotations

import numpy as np

from . import qm


def singlet_dm() -> np.ndarray:
    psi = qm.singlet()
    return np.outer(psi, psi.conj())


# --------------------------------------------------------------- C3: generator
def measurement_kraus(n: np.ndarray):
    """Kraus operators (P_+ (x) 1, P_- (x) 1) of the first measurement on A:
    supported on A's spin factor alone."""
    P = [qm.projector(n, s) for s in (+1, -1)]
    return [np.kron(p, np.eye(2)) for p in P]


def jump_update(rho: np.ndarray, n: np.ndarray, rng) -> tuple[int, np.ndarray]:
    """One realisation of the contact-point jump process: draw the outcome with
    the Born weight and apply the A-supported Kraus operator."""
    Ks = measurement_kraus(n)
    ps = [float(np.real(np.trace(K @ rho @ K.conj().T))) for K in Ks]
    i = 0 if rng.random() < ps[0] / (ps[0] + ps[1]) else 1
    K = Ks[i]
    out = K @ rho @ K.conj().T
    return (+1 if i == 0 else -1), out / np.real(np.trace(out))


def simulate_box(n_a: dict, n_b: dict, shots: int = 20000, seed: int = 2) -> dict:
    """CHSH box from the jump process (A first, then an ordinary measurement on
    the conditional state of B).  Should equal the quantum singlet box."""
    rng = np.random.default_rng(seed)
    E = {}
    for sa, na in n_a.items():
        for sb, nb in n_b.items():
            tot = 0.0
            for _ in range(shots // 4):
                a, rho1 = jump_update(singlet_dm(), na, rng)
                pb = float(np.real(np.trace(np.kron(np.eye(2), qm.projector(nb, +1)) @ rho1)))
                b = +1 if rng.random() < pb else -1
                tot += a * b
            E[(sa, sb)] = tot / (shots // 4)
    return E


def pushforward_is_projection(n: np.ndarray) -> float:
    """On the singlet, the A-supported operator P_s (x) 1 produces exactly the
    state the two-sided "joint projection" P_s (x) P_-s would — nothing need
    act on B for B to end up projected.  Returns the max deviation between the
    two post-measurement (unnormalised) states over both outcomes."""
    rho = singlet_dm()
    worst = 0.0
    for s in (+1, -1):
        KA = np.kron(qm.projector(n, s), np.eye(2))
        K2 = np.kron(qm.projector(n, s), qm.projector(n, -s))
        a = KA @ rho @ KA.conj().T
        b = K2 @ rho @ K2.conj().T
        worst = max(worst, float(np.max(np.abs(a - b))))
    return worst


def support_is_A_only(n: np.ndarray) -> bool:
    """Each Kraus operator is (operator on A) (x) identity on B."""
    for K in measurement_kraus(n):
        k = K.reshape(2, 2, 2, 2)   # (A_out, B_out, A_in, B_in)
        # K = a (x) 1  <=>  k[i, j, k_, l] = a[i, k_] delta[j, l]
        a = k[:, 0, :, 0]
        rebuilt = np.einsum("ik,jl->ijkl", a, np.eye(2))
        if not np.allclose(rebuilt, k, atol=1e-12):
            return False
    return True


# ------------------------------------------------------ C4: order independence
def realisable_A_first(P: np.ndarray, tol: float = 1e-9) -> bool:
    """P[a, b, s, s'] (indices 0/1 for outcomes +1/-1) admits the contact form
    P_A(a|s) P_B(b|s, a, s')  <=>  A's marginal ignores s'."""
    margA = P.sum(axis=1)          # [a, s, s']
    return bool(np.all(np.abs(margA[:, :, 0] - margA[:, :, 1]) < tol))


def realisable_B_first(P: np.ndarray, tol: float = 1e-9) -> bool:
    margB = P.sum(axis=0)          # [b, s, s']
    return bool(np.all(np.abs(margB[:, 0, :] - margB[:, 1, :]) < tol))


def order_independent_realisable(P: np.ndarray, tol: float = 1e-9) -> bool:
    """Realisable as a contact rule with either measurement first."""
    return realisable_A_first(P, tol) and realisable_B_first(P, tol)


def no_signalling(P: np.ndarray, tol: float = 1e-9) -> bool:
    return realisable_A_first(P, tol) and realisable_B_first(P, tol)


def pr_box() -> np.ndarray:
    """PR box: a (+) b (as bits) = s and s'."""
    P = np.zeros((2, 2, 2, 2))
    for s in (0, 1):
        for sp in (0, 1):
            for a in (0, 1):
                b = (a + (s & sp)) % 2
                P[a, b, s, sp] = 0.5
    return P


def one_way_signalling_box() -> np.ndarray:
    """b copies A's setting s: fine A-first (B may read s at the contact),
    impossible B-first."""
    P = np.zeros((2, 2, 2, 2))
    for s in (0, 1):
        for sp in (0, 1):
            for a in (0, 1):
                P[a, s, s, sp] = 0.5
    return P


def singlet_box(n_a: dict, n_b: dict) -> np.ndarray:
    P = np.zeros((2, 2, 2, 2))
    for si, na in enumerate(n_a.values()):
        for spi, nb in enumerate(n_b.values()):
            for ai, a in enumerate((+1, -1)):
                for bi, b in enumerate((+1, -1)):
                    Pa = np.kron(qm.projector(na, a), qm.projector(nb, b))
                    P[ai, bi, si, spi] = float(np.real(np.trace(Pa @ singlet_dm())))
    return P


# ----------------------------------------------------------- C5: QRF no-go
def chart_velocity_partner(rho: np.ndarray, v_rel: float = 1.0) -> float:
    """Velocity at which A's chart shows its partner: (1 - 2N(rho)) v_rel."""
    from . import r21_edge_weights as r21
    return (1.0 - r21.pair_weight_two_qubits(rho)) * v_rel


def werner_chart_velocity(p: float, v_rel: float = 1.0) -> float:
    return chart_velocity_partner(qm.werner(p), v_rel)


def affinity_defect(f, rho1: np.ndarray, rho2: np.ndarray, lams=None) -> float:
    """max over mixtures of |f(lam rho1 + (1-lam) rho2)
    - [lam f(rho1) + (1-lam) f(rho2)]| — zero for every affine functional."""
    lams = np.linspace(0, 1, 21) if lams is None else lams
    f1, f2 = f(rho1), f(rho2)
    return float(max(abs(f(l * rho1 + (1 - l) * rho2) - (l * f1 + (1 - l) * f2))
                     for l in lams))


def fixed_unitary_expectation(V: np.ndarray, O: np.ndarray):
    """The functional rho -> <O>_{V rho V^+}, affine by construction."""
    A = V.conj().T @ O @ V
    return lambda rho: float(np.real(np.trace(A @ rho)))
