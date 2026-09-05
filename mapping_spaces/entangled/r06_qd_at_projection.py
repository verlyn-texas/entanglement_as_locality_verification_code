"""R06 — what qD does at projection (M1a / M1b / M1c) and whether the
persistence of the label (M1c) carries a post-collapse *influence*.

Claims (numbered as in rows/R06_qd_at_projection.md):
 1. Projecting A along n1 leaves the pair in the product state |+n1>|-n1>
    (concurrence 0); second-round correlations factorise,
    E2(n2, n3) = (n1.n2)(-n1.n3), the same for either first-round outcome.
    With n1 = z, n2 = n3 = x: E2 = 0 and both second-round marginals vanish.
 2. M1c-influence (A's second measurement re-collapses B into |-s n2>):
    E2 = -n2.n3, independent of n1; with n1 = z, n2 = n3 = x: E2 = -1.
    Over a grid of settings the difference from QM is n2_perp.n3_perp
    (components perpendicular to n1): max 1 at n2 = n3 perp n1, zero when
    n2 || n1.
 3. The M1c-influence second-round box signals: B's marginal along z is
    P(-1) = 1 if A re-measures z and 1/2 if A measures x — signalling
    measure 0.5 in probability (expectation shift 1); the QM box has 0.
    The same box has CHSH S = 2 sqrt 2 on a product state; QM gives S = 0.
 4. QM's E2 = 0 for n2 perp n1 holds for *any* initial state (Werner,
    non-maximally entangled): after the projection A is in a pure n1
    eigenstate, so <n2.sigma>_A = n1.n2 = 0 whatever B's state.
 5. Trials to separate E2 = 0 from E2 = -1 at 5 sigma with readout fidelity
    F per measurement: N(F) = ceil(25 / (2F-1)^4) = 26 at F = 0.9994 (ions),
    39 at F = 0.95 (NV), 400 at F = 0.75.
 6. Kinematics: the lab tracks after projection are -y/a, +y/a under all
    three variants (beta_13(0) = beta_13(-1), beta_23(0) = beta_23(+1)); the
    variants differ only in the unobservable particle-frame separations.
 7. M1a, M1b and inert M1c give identical spin statistics (they share the
    projection rule): no pass-1 observable separates them.
"""
from __future__ import annotations

import math

import numpy as np

from mapping_spaces.entangled import frames, qm

# qD heights of (A, B) after the first projection, per variant.  Round-4
# note: the paper's canonical statement adopts M1c-inert (labels persist,
# edge weight 0) -- a label at z = 0 would contradict P1's single-valued
# lab potential for a moving particle; M1a/M1b are kept here only as the
# pass-1 alternatives whose spin statistics coincide (claim 7).
VARIANT_Z = {"M1a": (0.0, 0.0), "M1b": (0.0, 1.0), "M1c": (-1.0, 1.0)}


def _expect(rho1, n, which: str) -> float:
    return float(np.real(np.trace(qm.reduced(rho1, which) @ qm.spin_along(n))))


# ------------------------------------------------------------ claims 1, 2
def after_first_projection(n1, outcome: int = +1, state=None):
    """Pair state after A is projected along n1 with the given outcome."""
    state = qm.singlet() if state is None else state
    rho1, _ = qm.project(state, n1, outcome, "A")
    return rho1


def second_round_qm(n1, n2, n3, outcome: int = +1, state=None):
    """Standard QM (M1a, M1b, inert M1c): the second round measures the
    product state left by the first projection.  Returns (E2, <n2>_A, <n3>_B)."""
    rho1 = after_first_projection(n1, outcome, state)
    return (qm.correlation(rho1, n2, n3), _expect(rho1, n2, "A"), _expect(rho1, n3, "B"))


def second_round_influence(n1, n2, n3, outcome: int = +1, state=None):
    """M1c-influence: A's second measurement along n2 (outcome s) re-projects
    B into |-s n2>; B is then measured along n3.  Returns (E2, <n2>_A, <n3>_B)
    with the marginals averaged over s (B does not know s)."""
    rho1 = after_first_projection(n1, outcome, state)
    E2 = mA = mB = 0.0
    for s in (+1, -1):
        ps = qm.marginal_probability(rho1, n2, s, "A")
        rhoB = qm.projector(n2, -s)  # B re-collapsed, pure
        eB = float(np.real(np.trace(rhoB @ qm.spin_along(n3))))
        E2 += ps * s * eB
        mA += ps * s
        mB += ps * eB
    return E2, mA, mB


def closed_forms(n1, n2, n3):
    """Closed forms: QM (n1.n2)(-n1.n3); influence -n2.n3."""
    n1, n2, n3 = (np.asarray(v, dtype=float) for v in (n1, n2, n3))
    return float(np.dot(n1, n2) * (-np.dot(n1, n3))), float(-np.dot(n2, n3))


def e2_grid(n_theta: int = 13):
    """E2 under both models for n1 = z and n2, n3 in the x-z plane at polar
    angles theta2, theta3 on a grid.  Returns (thetas, E_qm, E_infl)."""
    thetas = np.linspace(0.0, np.pi, n_theta)
    n1 = qm.direction(0.0)
    E_qm = np.zeros((n_theta, n_theta))
    E_in = np.zeros((n_theta, n_theta))
    for i, t2 in enumerate(thetas):
        for j, t3 in enumerate(thetas):
            E_qm[i, j] = second_round_qm(n1, qm.direction(t2), qm.direction(t3))[0]
            E_in[i, j] = second_round_influence(n1, qm.direction(t2), qm.direction(t3))[0]
    return thetas, E_qm, E_in


def max_difference(n_theta: int = 13):
    thetas, E_qm, E_in = e2_grid(n_theta)
    diff = np.abs(E_in - E_qm)
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    return float(diff.max()), float(thetas[i]), float(thetas[j])


# ---------------------------------------------------------------- claim 3
def second_round_box(model: str, n1=None, settings_A=None, settings_B=None) -> np.ndarray:
    """Box P(a, b | x, y) for the second round, A measuring first.
    Default settings: A in {z, x}, B in {z, x}, first round along z."""
    n1 = qm.direction(0.0) if n1 is None else n1
    settings_A = settings_A or (qm.direction(0.0), qm.direction(np.pi / 2))
    settings_B = settings_B or (qm.direction(0.0), qm.direction(np.pi / 2))
    rho1 = after_first_projection(n1)
    if model == "qm":
        return qm.box_from_state(rho1, settings_A, settings_B)
    if model != "influence":
        raise ValueError(model)
    box = np.zeros((2, 2, 2, 2))
    for x, nA in enumerate(settings_A):
        for ia, s in enumerate(qm.OUTCOME):
            ps = qm.marginal_probability(rho1, nA, s, "A")
            rhoB = qm.projector(nA, -s)
            for y, nB in enumerate(settings_B):
                for ib, b in enumerate(qm.OUTCOME):
                    box[x, y, ia, ib] = ps * float(np.real(np.trace(rhoB @ qm.projector(nB, b))))
    return box


def signalling_and_chsh():
    """Signalling measure and CHSH of the second-round boxes.  CHSH uses A in
    {z, x} and B in {(z+x)/sqrt2, (z-x)/sqrt2}."""
    out = {}
    a, ap, b, bp = qm.chsh_optimal_settings()
    for model in ("qm", "influence"):
        box = second_round_box(model)
        pb = box.sum(axis=2)  # P(b | x, y)
        out[model] = {
            "signalling": qm.signalling(box),
            "P(b=-1|A=z,B=z)": float(pb[0, 0, 1]),
            "P(b=-1|A=x,B=z)": float(pb[1, 0, 1]),
            "chsh": qm.box_chsh(second_round_box(model, settings_A=(a, ap), settings_B=(b, bp))),
        }
    return out


# ---------------------------------------------------------------- claim 4
def robustness(ps=(1.0, 0.8, 0.5, 0.0), thetas=(np.pi / 4, 0.3, 0.1)):
    """E2 (QM) with n1 = z, n2 = n3 = x for Werner states and non-maximally
    entangled pure states."""
    z, x = qm.direction(0.0), qm.direction(np.pi / 2)
    rows = []
    for p in ps:
        rows.append(("werner", p, second_round_qm(z, x, x, state=qm.werner(p))[0]))
    for t in thetas:
        rows.append(("nonmax", t, second_round_qm(z, x, x, state=qm.nonmax_entangled(t))[0]))
    return rows


# ---------------------------------------------------------------- claim 5
def observed_correlation(E_true: float, F: float) -> float:
    """Each of the two readouts flips its outcome with probability 1 - F."""
    return (2 * F - 1) ** 2 * E_true


def trials_needed(F: float, n_sigma: float = 5.0, E_alt: float = -1.0) -> int:
    """Smallest N for which the observed correlations of the two models are
    n_sigma standard errors apart, the standard error taken under the wider
    (E = 0) hypothesis, sqrt((1 - E^2)/N) = 1/sqrt(N)."""
    dE = abs(observed_correlation(E_alt, F) - observed_correlation(0.0, F))
    if dE == 0:
        return math.inf
    return math.ceil((n_sigma / dE) ** 2)


# ---------------------------------------------------------------- claim 6
def post_projection_kinematics(a: float = 2.0, y: float = 1.0) -> dict:
    """Lab tracks and frame separations after the first projection, per
    variant, for a pair created at the origin (particles at rest in their
    own frames)."""
    out = {}
    for name, (zA, zB) in VARIANT_Z.items():
        xA = frames.beta(frames.A, frames.LAB, a, zA) * y
        xB = frames.beta(frames.B, frames.LAB, a, zB) * y
        out[name] = {
            "lab_tracks": (float(xA), float(xB)),
            "sep_lab": float(frames.separation_in_frame(frames.LAB, a, y, zA, zB)),
            "sep_frame_A": float(frames.separation_in_frame(frames.A, a, y, zA, zB)),
            "sep_frame_B": float(frames.separation_in_frame(frames.B, a, y, zA, zB)),
        }
    return out


# ---------------------------------------------------------------- claim 7
def variant_statistics(n1=None, n2=None, n3=None) -> dict:
    """Second-round (E2, mA, mB) per variant.  The qD rule never touches the
    spin state, so the three inert variants call the same projection."""
    n1 = qm.direction(0.0) if n1 is None else n1
    n2 = qm.direction(np.pi / 3) if n2 is None else n2
    n3 = qm.direction(2.0) if n3 is None else n3
    stats = {name: second_round_qm(n1, n2, n3) for name in VARIANT_Z}
    stats["M1c-influence"] = second_round_influence(n1, n2, n3)
    return stats


if __name__ == "__main__":
    z, x = qm.direction(0.0), qm.direction(np.pi / 2)
    print("claim 1  QM  E2, mA, mB (z; x, x):", second_round_qm(z, x, x), "outcome -1:", second_round_qm(z, x, x, -1))
    print("         concurrence after projection:", qm.concurrence(after_first_projection(z)))
    print("claim 2  influence E2, mA, mB (z; x, x):", second_round_influence(z, x, x))
    print("         max |diff| over grid, at (theta2, theta3):", max_difference())
    print("claim 3 ", signalling_and_chsh())
    print("claim 4 ", robustness())
    print("claim 5  N(F):", {F: trials_needed(F) for F in (0.9994, 0.95, 0.75)})
    print("claim 6 ", post_projection_kinematics())
    print("claim 7 ", variant_statistics())
