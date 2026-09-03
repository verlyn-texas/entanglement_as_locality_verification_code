"""R10 — the Cooper-pair splitter as the literal realisation of the mechanism
(hypothesis H10, mechanism M1a, experiment E10): a CHSH test with two
electrons created together in a superconductor, sent down two arms, and
spin-filtered at the arm ends.

Model dictionary (frames.py): the pair is created co-located at x = 0, y = 0
(feedback 7); the lab sees A at x = -y/a and B at x = +y/a; a filter at the
end of arm i acts at lab time y_i (= a L_i for arm length L_i).  In A's and
B's own frames both particles sit at the same x for all y, so each filter
acts on the pair *at the pair's own location*, the two filters being
separated only in y.  Spin statistics are those of R01 (M1a: singlet +
projection); this row adds the imperfections a real splitter has.

Claims (numbered as in rows/R10_cooper_pair_splitter.md):
 1. Geometry.  For arms of different lengths (y_A != y_B) the lab sees the
    measurement events at x = -y_A/a and +y_B/a, the pair's separation being
    2 y/a at each event; in frames A and B the two particles are at the same
    x at both measurement times.  Arms with unequal lab speeds v_A != v_B are
    the same model with a = 2/(v_A + v_B) and the common own-frame path
    p(y) = (v_B - v_A) y / 2 (the lab centre-of-mass drift): the lab tracks
    are recovered through frames.path_in_frame and co-location is exact.
 2. Imperfections.  With per-side spin-filter/detector fidelity F_A, F_B
    (an outcome is flipped with probability 1 - F), a fraction eta of
    coincidences that are genuine split pairs (the rest uncorrelated
    background, E = 0) and singlet fidelity p (Werner state),
        S_obs = 2 sqrt2 * p * eta * (2F_A - 1)(2F_B - 1).
    This equals the direct computation on the Werner box with the flip
    channel and background mixture applied, and the observed box is
    no-signalling.
 3. Thresholds.  S_obs > 2 iff p eta (2F - 1)^2 > 1/sqrt2 (symmetric F):
    F_min = 0.9204, 0.9432, 0.9701, 0.9855 for p eta = 1, 0.9, 0.8, 0.75;
    (p eta)_min = 0.7071, 0.8730 for F = 1, 0.95, and > 1 (impossible) for
    F = 0.92, 0.90.
 4. Statistics.  With N coincidences split equally over the four settings and
    Var(E_i) = (1 - E_i^2)/(N/4), a 5 sigma violation needs
    N_5sigma = 25 (16 - S_obs^2)/(S_obs - 2)^2 = 28975, 2975, 975, 445 for
    S_obs = 2.1, 2.3, 2.5, 2.7 (Monte Carlo confirms the variance formula).
 5. Ordering.  y is absolute, so which filter fires first (y_A < y_B or the
    reverse) is the same in the lab and in both particle frames, and the
    joint outcome distribution obtained by sequential projection in either
    order equals the box of the state (singlet and Werner): S does not
    depend on the order.
 6. Benchmarks and verdict.  L-A6d (co-located dots, S = 2.731) corresponds
    to an overall visibility V = S/2sqrt2 = 0.9656, i.e. an equivalent
    symmetric fidelity F = 0.9913 at p eta = 1; L-A4 (photons, S = 2.82759)
    to V = 0.99970.  The verdict rule: S_obs > 2 at >= 5 sigma -> consistent;
    S_obs <= 2 with p eta (2F-1)^2 > 1/sqrt2 -> the mechanism (and QM) fail
    in this geometry; otherwise inconclusive (visibility too low).
"""
from __future__ import annotations

import itertools
import numpy as np

from mapping_spaces.entangled import frames, qm

SQRT2 = np.sqrt(2.0)
LEDGER_S_A6D = 2.731   # Steinacker et al. 2025, direct parity readout (L-A6d)
LEDGER_S_A4 = 2.82759  # Poh et al. 2015 (L-A4)


# ------------------------------------------------------------- claim 1: geometry
def geometry(a: float, yA: float, yB: float, drift=None) -> dict:
    """Positions of A and B in the lab and in both particle frames at the two
    measurement times y_A, y_B.  ``drift`` is an optional common own-frame
    path p(y) (the lab centre-of-mass motion); default p = 0."""
    p = drift or (lambda t: 0.0 * t)
    out = {"yA": yA, "yB": yB, "first": frames.lab_time_order(yA, yB)}
    for k, name in ((frames.LAB, "lab"), (frames.A, "frame_A"), (frames.B, "frame_B")):
        for t, tname in ((yA, "at_yA"), (yB, "at_yB")):
            xA = float(frames.path_in_frame(frames.A, k, a, p, t))
            xB = float(frames.path_in_frame(frames.B, k, a, p, t))
            out[f"{name}_{tname}"] = (xA, xB)
            out[f"sep_{name}_{tname}"] = xB - xA
    return out


def asymmetric_arms(vA: float, vB: float, LA: float, LB: float) -> dict:
    """Arms with unequal lab speeds v_A, v_B (> 0) and lengths L_A, L_B.
    The model parameters are a = 2/(v_A + v_B) and the common own-frame path
    p(y) = (v_B - v_A) y / 2; the filters act at y_A = L_A/v_A, y_B = L_B/v_B."""
    a = 2.0 / (vA + vB)
    drift = lambda t: 0.5 * (vB - vA) * t  # noqa: E731
    yA, yB = LA / vA, LB / vB
    g = geometry(a, yA, yB, drift)
    # the lab tracks the model reproduces, versus the physical ones -v_A y, +v_B y
    ys = np.array([yA, yB])
    model_A = frames.path_in_frame(frames.A, frames.LAB, a, drift, ys)
    model_B = frames.path_in_frame(frames.B, frames.LAB, a, drift, ys)
    g.update({"a": a, "model_lab_A": model_A, "model_lab_B": model_B,
              "phys_lab_A": -vA * ys, "phys_lab_B": vB * ys})
    return g


# --------------------------------------------------------- claim 2: imperfections
def s_obs(p: float, eta: float, FA: float, FB: float) -> float:
    """Closed form S_obs = 2 sqrt2 p eta (2F_A - 1)(2F_B - 1)."""
    return float(qm.TSIRELSON * p * eta * (2 * FA - 1) * (2 * FB - 1))


def flip_channel(box: np.ndarray, FA: float, FB: float) -> np.ndarray:
    """Each wing's recorded outcome equals the true one with probability F
    and is flipped with probability 1 - F."""
    KA = np.array([[FA, 1 - FA], [1 - FA, FA]])
    KB = np.array([[FB, 1 - FB], [1 - FB, FB]])
    return np.einsum("ac,bd,xycd->xyab", KA, KB, box)


def background_box() -> np.ndarray:
    """Uncorrelated background coincidences: uniform outcomes, E = 0."""
    return np.full((2, 2, 2, 2), 0.25)


def observed_box(p: float, eta: float, FA: float, FB: float, settings=None) -> np.ndarray:
    """Werner(p) box at the CHSH settings, flip channel applied, mixed with a
    fraction 1 - eta of background."""
    a, ap, b, bp = settings or qm.chsh_optimal_settings()
    box = qm.box_from_state(qm.werner(p), (a, ap), (b, bp))
    return eta * flip_channel(box, FA, FB) + (1 - eta) * background_box()


def s_obs_direct(p: float, eta: float, FA: float, FB: float) -> float:
    """|S| of the observed box (the singlet's S at these settings is negative;
    S means |S| throughout this row)."""
    return float(abs(qm.box_chsh(observed_box(p, eta, FA, FB))))


def s_obs_check_grid() -> np.ndarray:
    """Max |closed form - direct| over a grid of (p, eta, F_A, F_B), and the
    max signalling of the observed boxes, as a length-2 array."""
    grid = itertools.product((1.0, 0.9, 0.75), (1.0, 0.8, 0.5), (1.0, 0.95, 0.9), (1.0, 0.97, 0.85))
    err, sig = 0.0, 0.0
    for p, eta, FA, FB in grid:
        box = observed_box(p, eta, FA, FB)
        err = max(err, abs(s_obs(p, eta, FA, FB) - abs(qm.box_chsh(box))))
        sig = max(sig, qm.signalling(box))
    return np.array([err, sig])


# ------------------------------------------------------------- claim 3: thresholds
VISIBILITY_THRESHOLD = 1 / SQRT2  # S_obs > 2  <=>  p eta (2F-1)^2 > 1/sqrt2


def f_min(p_eta: float) -> float:
    """Smallest symmetric fidelity with S_obs > 2 at overall pair quality p eta."""
    return float(0.5 * (1 + (SQRT2 * p_eta) ** -0.5))


def p_eta_min(F: float) -> float:
    """Smallest p eta with S_obs > 2 at symmetric fidelity F (> 1 means
    impossible at that fidelity)."""
    return float(1 / (SQRT2 * (2 * F - 1) ** 2))


def threshold_tables() -> tuple[dict, dict]:
    fm = {pe: f_min(pe) for pe in (1.0, 0.9, 0.8, 0.75)}
    pm = {F: p_eta_min(F) for F in (1.0, 0.95, 0.92, 0.90)}
    return fm, pm


# ------------------------------------------------------------- claim 4: statistics
def sigma_S(S: float, N: int) -> float:
    """Standard error of S when each of the four correlators |E_i| = S/4 is
    estimated from N/4 coincidences: Var S = sum_i (1 - E_i^2)/(N/4)."""
    return float(np.sqrt((16 - S * S) / N))


def n_5sigma(S: float, n_sigma: float = 5.0) -> int:
    """Coincidences needed for (S - 2)/sigma_S >= n_sigma."""
    if S <= 2:
        return 0  # no violation to detect
    return int(np.ceil(n_sigma ** 2 * (16 - S * S) / (S - 2) ** 2 - 1e-9))


def n_5sigma_table() -> dict:
    return {S: n_5sigma(S) for S in (2.1, 2.3, 2.5, 2.7)}


def mc_sigma_S(S: float, N: int, trials: int = 2000, seed: int = 10) -> tuple[float, float]:
    """Monte-Carlo (mean S_hat, std S_hat) from ``trials`` experiments of N
    coincidences drawn from an observed box with visibility S/2sqrt2."""
    rng = np.random.default_rng(seed)
    box = observed_box(S / qm.TSIRELSON, 1.0, 1.0, 1.0)
    n = N // 4
    signs = np.array([[1, -1], [1, 1]])
    S_hat = np.zeros(trials)
    for t in range(trials):
        total = 0.0
        for x in range(2):
            for y in range(2):
                counts = rng.multinomial(n, box[x, y].ravel())
                E = (counts[0] + counts[3] - counts[1] - counts[2]) / n
                total += signs[x, y] * E
        S_hat[t] = total
    return float(np.abs(S_hat).mean()), float(S_hat.std(ddof=1))


# --------------------------------------------------------------- claim 5: ordering
def sequential_joint(state, settings_A, settings_B, order: str) -> np.ndarray:
    """Joint distribution P(a,b|x,y) obtained by projecting the first particle
    (per ``order`` = "AB" or "BA"), then the second, with qm.project."""
    box = np.zeros((2, 2, 2, 2))
    for x, y, ia, ib in itertools.product(range(2), repeat=4):
        oa, ob = qm.OUTCOME[ia], qm.OUTCOME[ib]
        try:
            if order == "AB":
                rho1, p1 = qm.project(state, settings_A[x], oa, "A")
                _, p2 = qm.project(rho1, settings_B[y], ob, "B")
            else:
                rho1, p1 = qm.project(state, settings_B[y], ob, "B")
                _, p2 = qm.project(rho1, settings_A[x], oa, "A")
        except ValueError:  # first outcome impossible
            p1, p2 = 0.0, 0.0
        box[x, y, ia, ib] = p1 * p2
    return box


def order_independence(p: float = 1.0) -> dict:
    """Max |difference| between the A-first, B-first and box_from_state
    distributions for Werner(p), plus S from each."""
    a, ap, b, bp = qm.chsh_optimal_settings()
    rho = qm.werner(p)
    sA, sB = (a, ap), (b, bp)
    ab = sequential_joint(rho, sA, sB, "AB")
    ba = sequential_joint(rho, sA, sB, "BA")
    ref = qm.box_from_state(rho, sA, sB)
    return {"diff_AB": float(np.abs(ab - ref).max()), "diff_BA": float(np.abs(ba - ref).max()),
            "S_AB": float(abs(qm.box_chsh(ab))), "S_BA": float(abs(qm.box_chsh(ba))),
            "S_ref": float(abs(qm.box_chsh(ref)))}


def order_all_frames(a: float, yA: float, yB: float) -> dict:
    """Which measurement is first, read in the lab and in both particle frames
    after mapping the events (y is untouched by every map)."""
    xA, xB = frames.lab_tracks(a, np.array([yA, yB]))
    evA, evB = (xA[0], yA, frames.Z[frames.A]), (xB[1], yB, frames.Z[frames.B])
    out = {"lab": frames.lab_time_order(yA, yB)}
    for k, name in ((frames.A, "frame_A"), (frames.B, "frame_B")):
        _, tA, _ = frames.to_frame(frames.LAB, k, a, *evA)
        _, tB, _ = frames.to_frame(frames.LAB, k, a, *evB)
        out[name] = frames.lab_time_order(float(tA), float(tB))
    return out


# --------------------------------------------------- claim 6: benchmarks, verdict
def visibility_from_S(S: float) -> float:
    return float(S / qm.TSIRELSON)


def equivalent_symmetric_F(V: float) -> float:
    """Symmetric fidelity F with (2F - 1)^2 = V (p eta = 1)."""
    return float(0.5 * (1 + np.sqrt(V)))


def benchmarks() -> dict:
    V6d, V4 = visibility_from_S(LEDGER_S_A6D), visibility_from_S(LEDGER_S_A4)
    return {"V_A6d": V6d, "F_A6d": equivalent_symmetric_F(V6d), "V_A4": V4}


def verdict(S_meas: float, sigma: float, p: float, eta: float, F: float) -> str:
    """Decision rule of the row: what a measured S (+- sigma) means, given
    the independently calibrated p, eta, F."""
    predicted_violation = p * eta * (2 * F - 1) ** 2 > VISIBILITY_THRESHOLD
    if S_meas > 2 and (S_meas - 2) / sigma >= 5:
        return "consistent"
    if S_meas <= 2 and predicted_violation:
        return "mechanism fails"
    return "inconclusive"


if __name__ == "__main__":
    print("claim 1", geometry(2.0, 0.8, 1.2))
    print("claim 1 asym", asymmetric_arms(0.4, 0.6, 1.0, 0.9))
    print("claim 2", s_obs(0.9, 0.8, 0.95, 0.97), s_obs_direct(0.9, 0.8, 0.95, 0.97), s_obs_check_grid())
    print("claim 3", threshold_tables())
    print("claim 4", n_5sigma_table(), [(S, sigma_S(S, n_5sigma(S)), mc_sigma_S(S, n_5sigma(S))) for S in (2.3, 2.7)])
    print("claim 5", order_all_frames(2.0, 0.8, 1.2), order_independence(1.0), order_independence(0.8))
    print("claim 6", benchmarks(), [verdict(*args) for args in ((2.4, 0.05, 0.95, 0.9, 0.97), (1.9, 0.05, 0.95, 0.9, 0.97), (1.9, 0.05, 0.8, 0.8, 0.9))])
