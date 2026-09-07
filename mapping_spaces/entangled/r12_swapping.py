"""R12 — entanglement swapping, including delayed choice: how particles that
never met become "local", and what the locality narrative says when the swap
comes after the measurements.

Two pairs, A~B created at (x, y) = (0, 0) and C~D at (x0, 0); B and C are
brought to a common lab point at y = Y_BSM where a Bell-state measurement
(BSM) is made on them.  Qubit order everywhere is (A, B, C, D).

Claims (numbered as in rows/R12_swapping.md):
 1. Geometry (M10): before the BSM A~B and C~D are local and A, D are not;
    after it A~D are local and A~B, C~D are not (monogamy in the graph);
    with transfer = False (M10') nobody is local to A or D.  Positions are
    continuous through the event in every frame; the A-D separation in A's
    frame is constant after the BSM (6 lab units) while in the lab it grows
    as 2 + 2y; before the BSM it grows at the lab rate in every frame.
 2. Statistics: the four BSM outcomes have probability 1/4; the conditional
    A-D state has fidelity 1 with the Bell state (sigma_k x 1)|psi->; the
    Pauli correction sigma_k on D returns |psi->; corrected CHSH |S| = 2 sqrt 2
    for every outcome; the outcome-averaged (uncorrected) state is 1/4 and
    its CHSH is 0; A's and D's marginals are 1/2 whatever Victor does
    (BSM, SSM or nothing) — no-signalling.
 3. Order independence: the joint distribution P(s_A, t_D, k) is the same
    whether the BSM comes first (case a) or after A and D are measured
    (case b) — the projectors commute — so the post-selected A-D
    correlations and CHSH are identical.
 4. Narrative of case (b) in the model: A's projection leaves B in the pure
    state |-s a> and D's reduced state untouched (1/2); the BSM on the
    product |-s a>|-t d> has P(k | s, t) = |<beta_k | -s a, -t d>|^2, and
    (1/4) P(k | s, t) reproduces the case-(a) joint distribution exactly.
 5. Separable-state measurement (SSM, projection of B, C onto the
    computational basis): every conditional A-D state is a product state
    (concurrence 0, best Bell fidelity 1/2, witness W = 1/2 - F = 0) whereas
    the BSM gives W = -1/2 — the sign flip of L-D2.
 6. Graph in case (b): A and D are projected before the BSM, their edges are
    removed (postulate ii), the BSM finds B and C without partners and
    transfers nothing: A and D are never local.  In case (a) they are local
    on (Y_BSM, y_A) only.
 7. M10 vs M10' in case (a): identical statistics; the graphs differ only on
    (Y_BSM, y_A), where M10' has a concurrence-1 pair that is not local.
 8. Heights: four distinct heights (-1, 1, -2, 2) with four distinct lab
    velocities; the potentials are degree-4 polynomials in z (5 pinned
    heights); a 1-D qD suffices.
 9. Imperfect sources: two Werner sources p1, p2 give, after correction, a
    Werner swapped state with p = p1 p2 (visibility multiplies), so
    S = 2 sqrt 2 p1 p2.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import graph_frames as G, qm

# ------------------------------------------------------------ geometry
NAMES = ("A", "B", "C", "D")
HEIGHTS = {"A": -1.0, "B": 1.0, "C": -2.0, "D": 2.0}
VELOCITIES = {"A": -1.0, "B": 0.5, "C": -0.5, "D": 1.0}
X0 = {"A": 0.0, "B": 0.0, "C": 2.0, "D": 2.0}
Y_BSM = 2.0  # B (x = 0.5 y) and C (x = 2 - 0.5 y) meet at y = 2, x = 1
CASE_TIMES = {"a": {"bsm": Y_BSM, "A": 3.0, "D": 3.5}, "b": {"A": 1.0, "D": 1.5, "bsm": Y_BSM}}
PAIRS = (("A", "B"), ("C", "D"), ("A", "D"))


def swap_graph() -> G.LocalityGraph:
    g = G.LocalityGraph()
    for n in NAMES:
        g.add(G.Particle(n, HEIGHTS[n], VELOCITIES[n], X0[n]))
    return g


def history(case: str, transfer: bool = True) -> G.History:
    """Case a: entangle, BSM, then A and D measured.  Case b: entangle, A and
    D measured, then the BSM (delayed choice)."""
    t = CASE_TIMES[case]
    h = G.History(swap_graph())
    h.entangle("A", "B", 0.0).entangle("C", "D", 0.0)
    h.bell_measure("B", "C", t["bsm"], transfer=transfer)
    h.project("A", t["A"]).project("D", t["D"])
    return h


def locality_timeline(h: G.History, ys) -> dict:
    return {float(y): {i + j: h.graph_at(y).is_local(i, j) for i, j in PAIRS} for y in ys}


def meeting_point():
    """Lab (x, y) where B and C coincide."""
    yb = (X0["C"] - X0["B"]) / (VELOCITIES["B"] - VELOCITIES["C"])
    return float(X0["B"] + VELOCITIES["B"] * yb), float(yb)


def continuity_jumps(h: G.History, y0: float = Y_BSM, eps: float = 1e-3) -> float:
    """Largest |x(y0 + eps) - x(y0 - eps)| over all particles and frames."""
    ys = np.array([y0 - eps, y0 + eps])
    worst = 0.0
    for j in NAMES:
        for k in (*NAMES, G.LAB):
            x = h.position_in_frame(j, k, ys)
            worst = max(worst, abs(float(x[1] - x[0])))
    return worst


def ad_separations(h: G.History, ys):
    """(lab, A-frame, D-frame) separation x_D - x_A on a grid."""
    ys = np.asarray(ys, dtype=float)
    return {k: h.separation_in_frame("A", "D", k, ys) for k in (G.LAB, "A", "D")}


def interpolant_degree(g: G.LocalityGraph, k="A", zmax: float = 3.0) -> int:
    """Smallest polynomial degree that fits sigma_k(z) on a dense grid."""
    z = np.linspace(-zmax, zmax, 61)
    s = g.potential(k)(z)
    for deg in range(0, 9):
        coef = np.polyfit(z, s, deg)
        if np.max(np.abs(np.polyval(coef, z) - s)) < 1e-8:
            return deg
    raise RuntimeError("not polynomial of degree < 9")


# ------------------------------------------------------- four-qubit QM
PAULIS = (qm.I2, qm.SX, qm.SY, qm.SZ)
BELL_LABEL = ("psi-", "phi-", "phi+", "psi+")  # (sigma_k x 1)|psi-> up to phase


def initial_state() -> np.ndarray:
    """|psi->_AB (x) |psi->_CD, order (A, B, C, D)."""
    return np.kron(qm.singlet(), qm.singlet())


def bell_state(k: int) -> np.ndarray:
    return qm.kron(PAULIS[k], qm.I2) @ qm.singlet()


def bsm_projector(k: int) -> np.ndarray:
    """1_A (x) |beta_k><beta_k|_BC (x) 1_D."""
    return qm.kron(qm.I2, qm.dm(bell_state(k)), qm.I2)


def ssm_projector(m: int) -> np.ndarray:
    """1_A (x) |m><m|_BC (x) 1_D, m in {0: HH, 1: HV, 2: VH, 3: VV}."""
    e = np.zeros(4, dtype=complex)
    e[m] = 1.0
    return qm.kron(qm.I2, qm.dm(e), qm.I2)


def on_qubit(op: np.ndarray, which: str) -> np.ndarray:
    ops = [qm.I2] * 4
    ops[NAMES.index(which)] = op
    return qm.kron(*ops)


def reduced_pair(rho4: np.ndarray, keep=("A", "D")) -> np.ndarray:
    """Partial trace of a 4-qubit density matrix down to two qubits."""
    r = rho4.reshape([2] * 8)
    idx = [NAMES.index(k) for k in keep]
    drop = sorted(set(range(4)) - set(idx), reverse=True)
    for d in drop:
        r = np.trace(r, axis1=d, axis2=d + r.ndim // 2)
    n = 2 ** len(keep)
    return r.reshape(n, n)


def reduced_one(rho4: np.ndarray, which: str) -> np.ndarray:
    return reduced_pair(rho4, (which,))


def conditional_ad(state, proj: np.ndarray):
    """(rho_AD, probability) after the projector ``proj`` fires."""
    rho = qm.as_dm(state)
    prob = float(np.real(np.trace(proj @ rho)))
    return reduced_pair(proj @ rho @ proj / prob), prob


def fidelity(rho: np.ndarray, psi: np.ndarray) -> float:
    return float(np.real(psi.conj() @ rho @ psi))


def correct_on_d(rho_ad: np.ndarray, k: int) -> np.ndarray:
    U = qm.kron(qm.I2, PAULIS[k])
    return U @ rho_ad @ U.conj().T


def reflection(k: int) -> np.ndarray:
    """R_k d with sigma_k (d.sigma) sigma_k = (R_k d).sigma: identity for k=0,
    2 (d.e_k) e_k - d otherwise."""
    if k == 0:
        return np.eye(3)
    e = np.zeros(3)
    e[k - 1] = 1.0
    return 2 * np.outer(e, e) - np.eye(3)


def bsm_analysis(state=None) -> list:
    """Per outcome: probability, fidelity with beta_k on AD, corrected
    fidelity with |psi->, corrected |CHSH|."""
    state = initial_state() if state is None else state
    a, ap, b, bp = qm.chsh_optimal_settings()
    out = []
    for k in range(4):
        rho_k, p_k = conditional_ad(state, bsm_projector(k))
        corr = correct_on_d(rho_k, k)
        out.append({
            "k": k, "label": BELL_LABEL[k], "prob": p_k,
            "F_bell": fidelity(rho_k, bell_state(k)),
            "F_corrected": fidelity(corr, qm.singlet()),
            "chsh_corrected": abs(qm.chsh(corr, a, ap, b, bp)),
            "chsh_uncorrected": qm.chsh(rho_k, a, ap, b, bp),
            "concurrence": qm.concurrence(rho_k),
        })
    return out


def averaged_state(state=None) -> np.ndarray:
    state = initial_state() if state is None else state
    return sum(p * rho for rho, p in (conditional_ad(state, bsm_projector(k)) for k in range(4)))


def no_signalling_marginals(n=None) -> dict:
    """P(+1) for A and for D along n under: no Victor measurement, BSM
    (outcome unknown), SSM (outcome unknown)."""
    n = qm.direction(0.7, 0.3) if n is None else n
    rho = qm.dm(initial_state())
    out = {}
    for label, projs in (("none", [np.eye(16)]), ("bsm", [bsm_projector(k) for k in range(4)]),
                         ("ssm", [ssm_projector(m) for m in range(4)])):
        rho_v = sum(P @ rho @ P for P in projs)
        out[label] = {w: float(np.real(np.trace(rho_v @ on_qubit(qm.projector(n, +1), w)))) for w in ("A", "D")}
    return out


# ---------------------------------------------------- orderings (claims 3, 4)
def joint_case_a(a, d) -> np.ndarray:
    """P[k, s, t]: BSM first (outcome k), then A along a (s), D along d (t)."""
    J = np.zeros((4, 2, 2))
    for k in range(4):
        rho_k, p_k = conditional_ad(initial_state(), bsm_projector(k))
        for i_s, s in enumerate(qm.OUTCOME):
            for i_t, t in enumerate(qm.OUTCOME):
                J[k, i_s, i_t] = p_k * qm.joint_probability(rho_k, a, d, s, t)
    return J


def joint_case_b(a, d) -> np.ndarray:
    """P[k, s, t]: A and D projected first, then the BSM — computed as the
    expectation of the product of the three commuting projectors."""
    rho = qm.dm(initial_state())
    J = np.zeros((4, 2, 2))
    for k in range(4):
        for i_s, s in enumerate(qm.OUTCOME):
            for i_t, t in enumerate(qm.OUTCOME):
                op = on_qubit(qm.projector(a, s), "A") @ on_qubit(qm.projector(d, t), "D") @ bsm_projector(k)
                J[k, i_s, i_t] = float(np.real(np.trace(rho @ op)))
    return J


def commutator_norm(a, d) -> float:
    PA, PD = on_qubit(qm.projector(a, +1), "A"), on_qubit(qm.projector(d, +1), "D")
    worst = 0.0
    for k in range(4):
        Pk = bsm_projector(k)
        for P in (PA, PD):
            worst = max(worst, float(np.abs(P @ Pk - Pk @ P).max()))
    return worst


def narrative_case_b(a, d) -> dict:
    """The model's absolute-time story for case (b): A's projection (s)
    projects B into |-s a> (first-projection rule), D's (t) projects C into
    |-t d>; the BSM on the product state then has P(k | s, t).  Returns the
    joint (1/4) P(k|s,t) plus the checks on B's and D's reduced states."""
    rho = qm.dm(initial_state())
    J = np.zeros((4, 2, 2))
    checks = {}
    for i_s, s in enumerate(qm.OUTCOME):
        PA = on_qubit(qm.projector(a, s), "A")
        rho_s = PA @ rho @ PA / float(np.real(np.trace(PA @ rho)))
        rB, rD = reduced_one(rho_s, "B"), reduced_one(rho_s, "D")
        checks[s] = {"F_B(-s a)": float(np.real(np.trace(rB @ qm.projector(a, -s)))),
                     "D_max_dev_from_I/2": float(np.abs(rD - qm.I2 / 2).max())}
        for i_t, t in enumerate(qm.OUTCOME):
            chi = np.kron(_eigvec(a, -s), _eigvec(d, -t))  # B, C product state
            for k in range(4):
                J[k, i_s, i_t] = 0.25 * abs(bell_state(k).conj() @ chi) ** 2
    return {"joint": J, "checks": checks}


def _eigvec(n, outcome: int) -> np.ndarray:
    w, v = np.linalg.eigh(qm.spin_along(n))
    return v[:, int(np.argmin(np.abs(w - outcome)))]


def conditional_correlation(J: np.ndarray, k: int) -> float:
    Jk = J[k] / J[k].sum()
    return float(sum(Jk[i, j] * qm.OUTCOME[i] * qm.OUTCOME[j] for i in range(2) for j in range(2)))


def post_selected_chsh(joint_fn, k: int) -> float:
    """CHSH from post-selected correlations, D's settings classically
    corrected by R_k (Victor's record tells D which relabelling to apply)."""
    a, ap, b, bp = qm.chsh_optimal_settings()
    R = reflection(k)
    E = lambda x, y: conditional_correlation(joint_fn(x, R @ y), k)
    return abs(E(a, b) - E(a, bp) + E(ap, b) + E(ap, bp))


def narrative_case_a(a, d) -> dict:
    """Case (a): after the BSM (k) and correction, A's projection along a
    with outcome s leaves D pure in |-s a> — the pass-1 story, now for a
    pair that never met."""
    out = {}
    for k in range(4):
        rho_k, _ = conditional_ad(initial_state(), bsm_projector(k))
        corr = correct_on_d(rho_k, k)
        for s in qm.OUTCOME:
            rho_s, _ = qm.project(corr, a, s, "A")
            out[(k, int(s))] = float(np.real(np.trace(qm.reduced(rho_s, "B") @ qm.projector(a, -s))))
    return out


# ------------------------------------------------------------- SSM (claim 5)
def witness(rho_ad: np.ndarray) -> tuple[float, float]:
    """(W, F) with W = 1/2 - F, F = best fidelity with a Bell state."""
    F = max(fidelity(rho_ad, bell_state(k)) for k in range(4))
    return 0.5 - F, F


def ssm_analysis() -> list:
    out = []
    for m in range(4):
        rho_m, p_m = conditional_ad(initial_state(), ssm_projector(m))
        W, F = witness(rho_m)
        out.append({"m": m, "prob": p_m, "concurrence": qm.concurrence(rho_m), "F_best": F, "W": W})
    return out


def bsm_witnesses() -> list:
    return [witness(conditional_ad(initial_state(), bsm_projector(k))[0])[0] for k in range(4)]


# ------------------------------------------------- M10 vs M10' (claims 6, 7)
def graph_vs_state(case: str, transfer: bool, ys) -> list:
    """Per time: is A~D local in the graph, and what is the conditional A-D
    concurrence (1 after a BSM with no A/D measurement yet, else 0)."""
    h = history(case, transfer)
    t = CASE_TIMES[case]
    rows = []
    for y in ys:
        after_bsm = y > t["bsm"]
        a_measured, d_measured = y > t["A"], y > t["D"]
        if after_bsm and not (a_measured or d_measured):
            conc = qm.concurrence(conditional_ad(initial_state(), bsm_projector(0))[0])
        else:
            conc = 0.0
        rows.append({"y": float(y), "local_AD": h.graph_at(y).is_local("A", "D"), "concurrence_AD": conc})
    return rows


# ------------------------------------------------------- Werner (claim 9)
def werner_swap(p1: float, p2: float) -> dict:
    state = np.kron(qm.werner(p1), qm.werner(p2))
    out = {"p1p2": p1 * p2}
    a, ap, b, bp = qm.chsh_optimal_settings()
    for k in range(4):
        rho_k, p_k = conditional_ad(state, bsm_projector(k))
        corr = correct_on_d(rho_k, k)
        out[k] = {"prob": p_k, "dev_from_werner": float(np.abs(corr - qm.werner(p1 * p2)).max()),
                  "chsh": abs(qm.chsh(corr, a, ap, b, bp)), "visibility": -qm.correlation(corr, qm.direction(0), qm.direction(0))}
    return out


def visibility_from_chsh(S: float) -> float:
    return float(S / qm.TSIRELSON)


if __name__ == "__main__":
    ys = (0.5, 1.25, 1.75, 2.5, 3.25, 4.0)
    print("meeting point (x, y):", meeting_point())
    for tr in (True, False):
        print(f"claim 1/6 timeline case a transfer={tr}:", locality_timeline(history("a", tr), ys))
    print("claim 6 timeline case b:", locality_timeline(history("b"), ys))
    h = history("a")
    print("claim 1 max jump at BSM:", continuity_jumps(h))
    print("claim 1 A-D separations:", {k: v.round(6) for k, v in ad_separations(h, (0.0, 1.0, 2.0, 2.5, 3.0, 4.0)).items()})
    print("claim 8 interpolant degree:", interpolant_degree(swap_graph()))
    print("claim 2 BSM:", bsm_analysis())
    print("claim 2 averaged CHSH:", qm.chsh(averaged_state(), *qm.chsh_optimal_settings()), "marginals:", no_signalling_marginals())
    a, d = qm.direction(0.4, 0.2), qm.direction(1.9, -0.6)
    Ja, Jb, Jn = joint_case_a(a, d), joint_case_b(a, d), narrative_case_b(a, d)
    print("claim 3 |Ja - Jb|max:", np.abs(Ja - Jb).max(), "commutator:", commutator_norm(a, d))
    print("claim 4 |Ja - Jnarr|max:", np.abs(Ja - Jn["joint"]).max(), Jn["checks"])
    print("claim 3/4 post-selected CHSH:", [post_selected_chsh(joint_case_a, k) for k in range(4)],
          [post_selected_chsh(lambda x, y: narrative_case_b(x, y)["joint"], k) for k in range(4)])
    print("claim 4 case-a narrative:", narrative_case_a(a, d))
    print("claim 5 SSM:", ssm_analysis(), "BSM witnesses:", bsm_witnesses())
    print("claim 7 graph vs state (a):", graph_vs_state("a", True, ys), graph_vs_state("a", False, ys))
    print("claim 9 Werner:", werner_swap(0.9, 0.8), {S: visibility_from_chsh(S) for S in (2.421, 2.37, 2.38)})


# ------------------------------------------------- round-5 item T1 (A-N1)
def _dephase(rho: np.ndarray, projectors) -> np.ndarray:
    """Outcome-averaged (dephased) state after a projective measurement whose
    record is outside S: sum_k P_k rho P_k."""
    return sum(P @ rho @ P for P in projectors)


def _weights(rho: np.ndarray, n: int, pairs) -> dict:
    from mapping_spaces.entangled import r21_edge_weights as r21
    return {pair: float(r21.edge_weight(rho, pair[0], pair[1], n)) for pair in pairs}


def dephased_vs_conditional_weights() -> dict:
    """Round-5 item T1 (referee A-N1).  P2's edge weights read off the
    *dephased reduced* state of S (detector outside S) versus the
    *outcome-conditioned* state, for three measurements:

    * singlet, Z on A: w_AB = 0 both ways;
    * GHZ_3, X on particle 1: w_23 = 1 both ways;
    * swapping |psi->_AB |psi->_CD, Bell-basis measurement on B, C with
      S = {A, B, C, D}: dephased w_AD = w_BC = 0, conditional w_AD = w_BC = 1
      for every outcome.

    Single-particle projections agree; a Bell-basis projection on two
    partners does not.  The paper adopts the conditional-state rule, on which
    the swapped edge jumps from 0 to 1 at the projection."""
    out = {}
    # singlet + Z on A
    rho = qm.dm(qm.singlet())
    projs = [qm.kron(qm.projector(np.array([0, 0, 1.0]), s), qm.I2) for s in (+1, -1)]
    cond = [_weights(P @ rho @ P / np.real(np.trace(P @ rho)), 2, [(0, 1)])[(0, 1)] for P in projs]
    out["singlet_Z"] = {"dephased": _weights(_dephase(rho, projs), 2, [(0, 1)])[(0, 1)],
                        "conditional": cond}
    # GHZ + X on particle 1
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[7] = 1 / np.sqrt(2)
    rho = qm.dm(ghz)
    projs = [qm.kron(qm.projector(np.array([1.0, 0, 0]), s), qm.I2, qm.I2) for s in (+1, -1)]
    cond = [_weights(P @ rho @ P / np.real(np.trace(P @ rho)), 3, [(1, 2)])[(1, 2)] for P in projs]
    out["ghz_X"] = {"dephased": _weights(_dephase(rho, projs), 3, [(1, 2)])[(1, 2)],
                    "conditional": cond}
    # swapping: BSM on B, C
    rho = qm.dm(initial_state())
    projs = [bsm_projector(k) for k in range(4)]
    pairs = [(0, 3), (1, 2)]                       # A-D, B-C
    deph = _weights(_dephase(rho, projs), 4, pairs)
    cond = [_weights(P @ rho @ P / np.real(np.trace(P @ rho)), 4, pairs) for P in projs]
    out["swap_BSM"] = {"dephased": {"AD": deph[(0, 3)], "BC": deph[(1, 2)]},
                       "conditional": [{"AD": c[(0, 3)], "BC": c[(1, 2)]} for c in cond]}
    return out
