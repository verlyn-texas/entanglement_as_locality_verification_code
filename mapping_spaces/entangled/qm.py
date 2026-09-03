"""Two-qubit spin tools for the entangled-particles exploration (numpy).

Conventions
-----------
* Spin-1/2 operators are Pauli matrices; a measurement "along n" is n.sigma
  with outcomes +-1.  Directions are unit 3-vectors; ``direction(theta, phi)``
  builds one, and ``direction(theta)`` lies in the x-z plane.
* Two-qubit operators act on C^2 (x) C^2 with particle A first.
* CHSH: S = E(a,b) - E(a,b') + E(a',b) + E(a',b').  Local realism: |S| <= 2.
  Quantum (Tsirelson): |S| <= 2 sqrt 2.  No-signalling polytope: |S| <= 4.
"""
from __future__ import annotations

import itertools
import numpy as np

# ---------------------------------------------------------------- operators
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = np.stack([SX, SY, SZ])

TSIRELSON = 2 * np.sqrt(2)


def direction(theta: float, phi: float = 0.0) -> np.ndarray:
    """Unit vector with polar angle theta from +z and azimuth phi."""
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def spin_along(n) -> np.ndarray:
    """n . sigma for a unit 3-vector n."""
    n = np.asarray(n, dtype=float)
    return np.einsum("i,ijk->jk", n, PAULI)


def projector(n, outcome: int) -> np.ndarray:
    """Projector onto the eigenvalue ``outcome`` (+1/-1) of n.sigma."""
    return 0.5 * (I2 + outcome * spin_along(n))


def kron(*ops) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


# ------------------------------------------------------------------- states
def ket(*amps) -> np.ndarray:
    v = np.array(amps, dtype=complex)
    return v / np.linalg.norm(v)


def singlet() -> np.ndarray:
    """|psi-> = (|01> - |10>)/sqrt2 as a state vector."""
    return ket(0, 1, -1, 0)


def dm(psi: np.ndarray) -> np.ndarray:
    """Density matrix of a pure state vector."""
    psi = np.asarray(psi, dtype=complex).reshape(-1, 1)
    return psi @ psi.conj().T


def werner(p: float) -> np.ndarray:
    """Werner state p|psi-><psi-| + (1-p) I/4 (a partially decohered singlet)."""
    return p * dm(singlet()) + (1 - p) * np.eye(4) / 4


def nonmax_entangled(theta: float) -> np.ndarray:
    """cos(theta)|01> - sin(theta)|10>; theta = pi/4 is the singlet."""
    return ket(0, np.cos(theta), -np.sin(theta), 0)


def as_dm(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex)
    return dm(state) if state.ndim == 1 else state


# ------------------------------------------------------------- statistics
def correlation(state, a, b) -> float:
    """E(a,b) = <(a.sigma)(x)(b.sigma)>."""
    rho = as_dm(state)
    return float(np.real(np.trace(rho @ kron(spin_along(a), spin_along(b)))))


def joint_probability(state, a, b, oa: int, ob: int) -> float:
    rho = as_dm(state)
    return float(np.real(np.trace(rho @ kron(projector(a, oa), projector(b, ob)))))


def marginal_probability(state, n, outcome: int, which: str) -> float:
    rho = as_dm(state)
    op = kron(projector(n, outcome), I2) if which == "A" else kron(I2, projector(n, outcome))
    return float(np.real(np.trace(rho @ op)))


def chsh(state, a, ap, b, bp) -> float:
    return (correlation(state, a, b) - correlation(state, a, bp)
            + correlation(state, ap, b) + correlation(state, ap, bp))


def chsh_optimal_settings():
    """Settings in the x-z plane that maximise |S| for the singlet."""
    return (direction(0.0), direction(np.pi / 2), direction(np.pi / 4), direction(3 * np.pi / 4))


def chsh_max(state) -> float:
    """Horodecki bound: max |S| = 2 sqrt(m1 + m2), m_i the two largest
    eigenvalues of T^T T for the correlation matrix T_ij = <sigma_i x sigma_j>."""
    rho = as_dm(state)
    T = np.array([[np.real(np.trace(rho @ kron(PAULI[i], PAULI[j]))) for j in range(3)] for i in range(3)])
    ev = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
    return float(2 * np.sqrt(ev[0] + ev[1]))


def concurrence(state) -> float:
    """Wootters concurrence of a two-qubit state."""
    rho = as_dm(state)
    yy = kron(SY, SY)
    rt = rho @ yy @ rho.conj() @ yy
    ev = np.sort(np.sqrt(np.abs(np.linalg.eigvals(rt))))[::-1]
    return float(max(0.0, ev[0] - ev[1] - ev[2] - ev[3]))


def project(state, n, outcome: int, which: str) -> tuple[np.ndarray, float]:
    """Projective (collapse) measurement on one particle.  Returns the
    normalised post-measurement density matrix and the outcome probability."""
    rho = as_dm(state)
    P = kron(projector(n, outcome), I2) if which == "A" else kron(I2, projector(n, outcome))
    prob = float(np.real(np.trace(P @ rho)))
    if prob == 0:
        raise ValueError("outcome has zero probability")
    return P @ rho @ P / prob, prob


def reduced(state, which: str) -> np.ndarray:
    """Reduced density matrix of one particle."""
    rho = as_dm(state).reshape(2, 2, 2, 2)
    return np.trace(rho, axis1=1, axis2=3) if which == "A" else np.trace(rho, axis1=0, axis2=2)


# ---------------------------------------------- boxes P(a,b|x,y), x,y in {0,1}
# A "box" is an array box[x, y, a, b] with a,b indexed 0 -> +1, 1 -> -1.
OUTCOME = np.array([1, -1])


def box_from_state(state, settings_A, settings_B) -> np.ndarray:
    box = np.zeros((2, 2, 2, 2))
    for x, y, ia, ib in itertools.product(range(2), repeat=4):
        box[x, y, ia, ib] = joint_probability(state, settings_A[x], settings_B[y], OUTCOME[ia], OUTCOME[ib])
    return box


def box_correlation(box, x, y) -> float:
    return float(sum(box[x, y, ia, ib] * OUTCOME[ia] * OUTCOME[ib] for ia in range(2) for ib in range(2)))


def box_chsh(box) -> float:
    E = box_correlation
    return E(box, 0, 0) - E(box, 0, 1) + E(box, 1, 0) + E(box, 1, 1)


def signalling(box) -> float:
    """Largest change of a wing's marginal when the *other* wing's setting
    changes. Zero for a no-signalling box."""
    pa = box.sum(axis=3)  # P(a|x,y)
    pb = box.sum(axis=2)  # P(b|x,y)
    return float(max(np.abs(pa[:, 0, :] - pa[:, 1, :]).max(), np.abs(pb[0, :, :] - pb[1, :, :]).max()))


def pr_box() -> np.ndarray:
    """Popescu-Rohrlich box: outcomes anti-correlated only for the (x,y)=(0,1)
    pair (the term that enters CHSH with a minus sign here), so S = 4;
    marginals are uniform, so it is no-signalling."""
    box = np.zeros((2, 2, 2, 2))
    for x, y, ia, ib in itertools.product(range(2), repeat=4):
        if (ia ^ ib) == (1 if (x, y) == (0, 1) else 0):
            box[x, y, ia, ib] = 0.5
    return box


def contact_box(pa, pb_given) -> np.ndarray:
    """Box realised by a *sequential contact* model: A answers first with
    P(a|x) = pa[x, a]; B, co-located with A, answers with P(b|x,a,y) =
    pb_given[x, a, y, b]."""
    box = np.zeros((2, 2, 2, 2))
    for x, y, ia, ib in itertools.product(range(2), repeat=4):
        box[x, y, ia, ib] = pa[x, ia] * pb_given[x, ia, y, ib]
    return box


def deterministic_contact_boxes():
    """All boxes from deterministic contact strategies: f: x -> a (4 maps)
    and g: (x, a, y) -> b (256 maps). Yields (f, g, box)."""
    for f in itertools.product(range(2), repeat=2):
        pa = np.zeros((2, 2))
        for x in range(2):
            pa[x, f[x]] = 1.0
        for g in itertools.product(range(2), repeat=8):
            pb = np.zeros((2, 2, 2, 2))
            for x, ia, y in itertools.product(range(2), repeat=3):
                pb[x, ia, y, g[4 * x + 2 * ia + y]] = 1.0
            yield f, g, contact_box(pa, pb)


def qm_conditional_contact(state, settings_A, settings_B) -> np.ndarray:
    """The contact model tuned to quantum mechanics: A draws from the QM
    marginal, B draws from the QM conditional given (x, a, y)."""
    qbox = box_from_state(state, settings_A, settings_B)
    pa = qbox.sum(axis=3)[:, 0, :]  # P(a|x) (y-independent for a QM box)
    pb = np.zeros((2, 2, 2, 2))
    for x, ia, y in itertools.product(range(2), repeat=3):
        pb[x, ia, y, :] = qbox[x, y, ia, :] / pa[x, ia]
    return contact_box(pa, pb)
