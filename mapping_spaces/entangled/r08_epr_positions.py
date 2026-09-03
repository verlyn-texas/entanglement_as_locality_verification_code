"""R08 — mechanism M6: position–momentum (EPR) entanglement of the pair's
motion along x, with the particles' common frame read as the
relative-coordinate frame.

Conventions (kept throughout)
-----------------------------
* Phase-space vector R = (x1, p1, x2, p2); covariance V_ij = <{dR_i, dR_j}>/2.
* SI-like units with hbar explicit: the *reference* single-particle packet
  has position variance sigma0^2 and momentum variance sigma_p^2 = hbar^2 /
  (4 sigma0^2), i.e. a minimum-uncertainty packet, sigma0 sigma_p = hbar/2.
  Setting hbar = 1, sigma0 = 1/sqrt2 gives the dimensionless quadratures
  ([X, P] = i, vacuum variance 1/2) used in the CV literature.
* The two-mode squeezed vacuum (TMSV) with parameter r >= 0 has
      Var(x1 - x2) = 2 sigma0^2 e^{-2r},  Var(p1 + p2) = 2 sigma_p^2 e^{-2r},
      Var(x1 + x2) = 2 sigma0^2 e^{+2r},  Var(p1 - p2) = 2 sigma_p^2 e^{+2r},
  i.e. correlated positions and anti-correlated momenta (EPR 1935).
* Lab = frame 3; A = object 1 (z = -1), B = object 2 (z = +1); y = lab time.

Claims (numbered as in rows/R08_epr_positions.md):
 1. TMSV covariance matrix: the four variances above, single-particle
    Var(x_i) = sigma0^2 cosh 2r, and V is a bona fide quantum state
    (symplectic eigenvalues hbar/2).
 2. Reid criterion: Var_inf(x1|x2) Var_inf(p1|p2) = (hbar/2)^2 / cosh^2(2r)
    < (hbar/2)^2 for every r > 0; ratios 0.4200, 0.07065, 0.001341 at r = 0.5, 1, 2.
 3. Duan–Simon: normalised sum 2 e^{-2r} < 2 (0.7358, 0.2707, 0.03663), and
    the PPT symplectic eigenvalue nu_- = (hbar/2) e^{-2r} < hbar/2.
 4. Frame dictionary: the qD frame maps act on the phase-space *mean* only
    (a z-dependent Galilean displacement) and leave V invariant; in frames 1
    and 2 the mean of x1 - x2 is 0 for all y and its rms is sqrt2 sigma0 e^{-r};
    the Gaussian means reproduce frames.lab_tracks.
 5. Free evolution: V(t) = S V S^T; Var(x_rel(t)) = 2 sigma0^2 e^{-2r}
    + 2 sigma_p^2 e^{2r} t^2 / m^2 (cross term zero); Var(p1 +- p2),
    Var(x1 - x2 - (p1 - p2) t/m), the symplectic eigenvalues and nu_- are
    constant.
 6. Electron numbers (sigma0 = 10 nm, r = 1): rms co-location 5.20 nm;
    doubling time t2 = 2 sqrt3 m sigma0^2 e^{-2r}/hbar = 4.05e-13 s
    (e^{2r} = 7.39 times shorter than for r = 0); fixed-quadrature Duan
    witness lost at 8.36e-13 s, fixed-quadrature Reid witness lost at a
    computed t_R, while nu_- never changes; single-particle uncertainty
    product >= hbar^2/4 at all t; sigma0 = 0.50 um gives t2 = 1 ns.
 7. Spin (x) motion: CHSH = 2 sqrt2 for every r; a spin projection leaves
    (mean, V) unchanged; the Reid product is unchanged by spin measurement.
 8. Ledger: M6 predicts standard QM for L-C3a/b/c (no new numbers); read as
    a Gaussian, an EPR ratio 0.72 corresponds to r_eff = 0.294 (illustrative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mapping_spaces.entangled import frames, qm

# ------------------------------------------------------------- constants (SI)
HBAR = 1.054571817e-34      # J s
M_E = 9.1093837015e-31      # kg
M_HE4 = 6.6464731e-27       # kg (illustration only)

# combination vectors in R = (x1, p1, x2, p2)
X_MINUS = np.array([1.0, 0.0, -1.0, 0.0])
X_PLUS = np.array([1.0, 0.0, 1.0, 0.0])
P_PLUS = np.array([0.0, 1.0, 0.0, 1.0])
P_MINUS = np.array([0.0, 1.0, 0.0, -1.0])

OMEGA = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)


# ------------------------------------------------------------------- state
def sigma_p(sigma0: float, hbar: float = HBAR) -> float:
    """Momentum width of the minimum-uncertainty reference packet."""
    return hbar / (2.0 * sigma0)


def tmsv_cov(r: float, sigma0: float = 1 / np.sqrt(2), hbar: float = 1.0) -> np.ndarray:
    """Covariance matrix of the two-mode squeezed vacuum in (x1,p1,x2,p2)."""
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    v = 0.5 * np.array([[c, 0, s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, -s, 0, c]])
    ell = np.sqrt(2.0) * sigma0                # vacuum Var(x) = ell^2/2 = sigma0^2
    d = np.diag([ell, hbar / ell, ell, hbar / ell])
    return d @ v @ d


def var_of(V: np.ndarray, c: np.ndarray) -> float:
    """Variance of the linear combination c . R."""
    return float(c @ V @ c)


def cov_of(V: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> float:
    return float(c1 @ V @ c2)


def symplectic_eigenvalues(V: np.ndarray) -> np.ndarray:
    """Symplectic eigenvalues (>= hbar/2 for a physical state)."""
    ev = np.linalg.eigvals(1j * OMEGA @ V)
    return np.sort(np.abs(ev))[::2]


def ppt_min_symplectic(V: np.ndarray) -> float:
    """Smallest symplectic eigenvalue of the partially transposed V
    (p2 -> -p2).  < hbar/2 iff the two-mode Gaussian state is entangled
    (Simon 2000)."""
    lam = np.diag([1.0, 1.0, 1.0, -1.0])
    return float(symplectic_eigenvalues(lam @ V @ lam)[0])


# --------------------------------------------------------------- criteria
def conditional_variance(V: np.ndarray, i: int, j: int) -> float:
    """Var_inf(R_i | R_j) = Var(R_i) - Cov(R_i,R_j)^2 / Var(R_j)."""
    return float(V[i, i] - V[i, j] ** 2 / V[j, j])


def reid(V: np.ndarray, hbar: float = 1.0) -> dict:
    """Reid EPR criterion with the fixed quadratures x, p:
    Var_inf(x1|x2) Var_inf(p1|p2) < (hbar/2)^2."""
    vx = conditional_variance(V, 0, 2)
    vp = conditional_variance(V, 1, 3)
    prod = vx * vp
    return {"var_inf_x": vx, "var_inf_p": vp, "product": prod,
            "ratio": prod / (hbar / 2) ** 2, "satisfied": prod < (hbar / 2) ** 2}


def duan(V: np.ndarray, sigma0: float, hbar: float = 1.0) -> dict:
    """Duan–Simon sum with the fixed quadratures, normalised so that every
    separable state has sum >= 2:  Var(x1-x2)/ell^2 + Var(p1+p2)/(hbar/ell)^2,
    ell^2 = 2 sigma0^2."""
    ell2 = 2.0 * sigma0 ** 2
    s = var_of(V, X_MINUS) / ell2 + var_of(V, P_PLUS) / (hbar ** 2 / ell2)
    return {"sum": s, "ratio": s / 2.0, "satisfied": s < 2.0}


# ------------------------------------------------------ frame dictionary (4)
def lab_mean(a: float, m: float, y: float) -> np.ndarray:
    """Phase-space mean in the lab: A at -y/a with <p> = -m/a, B at +y/a
    with <p> = +m/a (the pair was created at x = 0 at y = 0)."""
    return np.array([-y / a, -m / a, y / a, m / a])


def mean_in_frame(mean_lab: np.ndarray, k: int, a: float, m: float, y: float) -> np.ndarray:
    """The qD frame map T_{3k} as a z-dependent Galilean displacement:
    x_i -> x_i + beta_{3k}(z_i) y,  p_i -> p_i + m beta_{3k}(z_i)."""
    bA = float(frames.beta(frames.LAB, k, a, frames.Z[frames.A]))
    bB = float(frames.beta(frames.LAB, k, a, frames.Z[frames.B]))
    return mean_lab + np.array([bA * y, m * bA, bB * y, m * bB])


def cov_in_frame(V: np.ndarray, k: int) -> np.ndarray:
    """A displacement leaves the covariance matrix unchanged."""
    return V.copy()


def colocation_rms(r: float, sigma0: float) -> float:
    """rms of x_A - x_B in the particles' frame: sqrt(2) sigma0 e^{-r}."""
    return float(np.sqrt(2.0) * sigma0 * np.exp(-r))


# --------------------------------------------------------- free flight (5, 6)
def free_symplectic(t: float, m: float) -> np.ndarray:
    S = np.eye(4)
    S[0, 1] = S[2, 3] = t / m
    return S


def evolve(V: np.ndarray, mean: np.ndarray, t: float, m: float):
    S = free_symplectic(t, m)
    return S @ V @ S.T, S @ mean


def x_rel_variance_closed_form(t, r: float, sigma0: float, m: float, hbar: float = HBAR):
    t = np.asarray(t, dtype=float)
    sp = sigma_p(sigma0, hbar)
    return 2 * sigma0 ** 2 * np.exp(-2 * r) + 2 * sp ** 2 * np.exp(2 * r) * t ** 2 / m ** 2


def back_evolved_x_minus(t: float, m: float) -> np.ndarray:
    """Combination x1 - x2 - (p1 - p2) t/m, whose variance is constant."""
    return X_MINUS - (t / m) * P_MINUS


def doubling_time(r: float, sigma0: float, m: float, hbar: float = HBAR) -> float:
    """Time at which sqrt Var(x_rel) has doubled: 2 sqrt3 m sigma0^2 e^{-2r}/hbar."""
    return float(2 * np.sqrt(3.0) * m * sigma0 ** 2 * np.exp(-2 * r) / hbar)


def sigma0_for_doubling_time(t2: float, r: float, m: float, hbar: float = HBAR) -> float:
    return float(np.sqrt(t2 * hbar * np.exp(2 * r) / (2 * np.sqrt(3.0) * m)))


def duan_loss_time(r: float, sigma0: float, m: float, hbar: float = HBAR) -> float:
    """Time at which the fixed-quadrature Duan sum reaches 2:
    (2 m sigma0^2/hbar) e^{-r} sqrt(2 - 2 e^{-2r})."""
    return float(2 * m * sigma0 ** 2 / hbar * np.exp(-r) * np.sqrt(2 - 2 * np.exp(-2 * r)))


def _bisect(f, lo: float, hi: float, n: int = 200) -> float:
    flo = f(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reid_loss_time(r: float, sigma0: float, m: float, hbar: float = HBAR) -> float:
    """Time at which the fixed-quadrature Reid product reaches (hbar/2)^2."""
    V0 = tmsv_cov(r, sigma0, hbar)

    def f(t):
        return reid(evolve(V0, np.zeros(4), t, m)[0], hbar)["ratio"] - 1.0

    hi = 1.0
    while f(hi) < 0:
        hi *= 10
    return _bisect(f, 0.0, hi)


def free_flight_table(r: float, sigma0: float, m: float, times, hbar: float = HBAR) -> dict:
    """Everything claim 5/6 needs along a list of times."""
    V0 = tmsv_cov(r, sigma0, hbar)
    out = {k: [] for k in ("t", "var_x_rel", "var_x_rel_closed", "var_p_plus", "var_p_minus",
                            "var_back_evolved", "nu_minus", "sympl_min", "duan_sum",
                            "reid_ratio", "uncert_1", "cov_xrel_prel")}
    for t in times:
        V, _ = evolve(V0, np.zeros(4), t, m)
        out["t"].append(t)
        out["var_x_rel"].append(var_of(V, X_MINUS))
        out["var_x_rel_closed"].append(float(x_rel_variance_closed_form(t, r, sigma0, m, hbar)))
        out["var_p_plus"].append(var_of(V, P_PLUS))
        out["var_p_minus"].append(var_of(V, P_MINUS))
        out["var_back_evolved"].append(var_of(V, back_evolved_x_minus(t, m)))
        out["nu_minus"].append(ppt_min_symplectic(V))
        out["sympl_min"].append(float(symplectic_eigenvalues(V)[0]))
        out["duan_sum"].append(duan(V, sigma0, hbar)["sum"])
        out["reid_ratio"].append(reid(V, hbar)["ratio"])
        out["uncert_1"].append(float(V[0, 0] * V[1, 1]))
        out["cov_xrel_prel"].append(cov_of(V, X_MINUS, 0.5 * P_MINUS))
    return {k: np.array(v) for k, v in out.items()}


# -------------------------------------------------------- spin (x) motion (7)
@dataclass
class PairState:
    """Spin density matrix (x) motional Gaussian (mean, cov).  The tensor
    product is the M6 assumption: the entangling event fixes the spin
    singlet and the motional Gaussian independently (no spin–orbit term)."""
    rho_spin: np.ndarray
    mean: np.ndarray
    cov: np.ndarray


def make_pair(r: float, sigma0: float, hbar: float, a: float, m: float, y: float = 0.0) -> PairState:
    return PairState(qm.dm(qm.singlet()), lab_mean(a, m, y), tmsv_cov(r, sigma0, hbar))


def chsh_of(state: PairState) -> float:
    a, ap, b, bp = qm.chsh_optimal_settings()
    return float(qm.chsh(state.rho_spin, a, ap, b, bp))


def project_spin(state: PairState, n, outcome: int, which: str) -> tuple[PairState, float]:
    """Projective spin measurement on one particle: acts on the spin factor
    only; (mean, cov) are carried over unchanged."""
    rho, p = qm.project(state.rho_spin, n, outcome, which)
    return PairState(rho, state.mean.copy(), state.cov.copy()), p


# ------------------------------------------------------------------ ledger (8)
def effective_r_from_reid_ratio(ratio: float) -> float:
    """Invert ratio = 1/cosh^2(2r) (illustrative Gaussian reading)."""
    return float(0.5 * np.arccosh(1.0 / np.sqrt(ratio)))


def ledger_entries() -> dict:
    """Numbers from registers/constraints.md (verified 2026-08-25) with the
    M6 prediction: standard QM, no new numbers."""
    return {
        "L-C3a": {"E": 0.51, "dE": 0.20, "qm_bound": 1.0, "m6": "QM"},
        "L-C3b": {"S": 1.77, "dS": 0.06, "classical_bound": np.sqrt(2), "qm_bound": 2.0, "m6": "QM"},
        "L-C3c": {"epr": 0.18, "depr": 0.03, "bound": 0.25, "m6": "QM",
                  "r_eff_illustrative": effective_r_from_reid_ratio(0.18 / 0.25)},
        "L-C3d": {"electron_pairs": "none found", "grade": "V2"},
    }


# ------------------------------------------------------------------ summary
def summary(sigma0: float = 10e-9, r: float = 1.0, m: float = M_E, hbar: float = HBAR,
            a: float = 2.0) -> dict:
    out = {}
    # 1–3 dimensionless
    for rr in (0.0, 0.5, 1.0, 2.0):
        V = tmsv_cov(rr)
        out[f"reid_ratio_r{rr}"] = reid(V)["ratio"]
        out[f"duan_sum_r{rr}"] = duan(V, 1 / np.sqrt(2))["sum"]
        out[f"nu_minus_r{rr}"] = ppt_min_symplectic(V)
    # 4 frames
    out["colocation_rms"] = colocation_rms(r, sigma0)
    # 6 electron numbers
    out["t2"] = doubling_time(r, sigma0, m, hbar)
    out["t2_r0"] = doubling_time(0.0, sigma0, m, hbar)
    out["t_duan"] = duan_loss_time(r, sigma0, m, hbar)
    out["t_reid"] = reid_loss_time(r, sigma0, m, hbar)
    out["sigma0_1ns"] = sigma0_for_doubling_time(1e-9, r, m, hbar)
    out["t2_he4_1um"] = doubling_time(r, 1e-6, M_HE4, hbar)
    # 7 spin
    st = make_pair(r, sigma0, hbar, a, m)
    out["chsh"] = chsh_of(st)
    st2, _ = project_spin(st, qm.direction(0.7), +1, "A")
    out["cov_unchanged_by_spin_projection"] = bool(np.array_equal(st.cov, st2.cov))
    out["ledger"] = ledger_entries()
    return out


if __name__ == "__main__":
    s = summary()
    for k, v in s.items():
        print(f"{k}: {v}")
    ts = np.array([0, 0.25, 0.5, 1, 2, 4]) * s["t2"]
    tab = free_flight_table(1.0, 10e-9, M_E, ts)
    print("t [s]        rms x_rel [nm]  Var(p1+p2)/Var0  back-evolved  nu_-/(hbar/2)  Duan  Reid")
    for i in range(len(ts)):
        print(f"{tab['t'][i]:.3e}  {np.sqrt(tab['var_x_rel'][i]) * 1e9:8.3f}  "
              f"{tab['var_p_plus'][i] / tab['var_p_plus'][0]:.6f}  "
              f"{tab['var_back_evolved'][i] / tab['var_back_evolved'][0]:.6f}  "
              f"{tab['nu_minus'][i] / (HBAR / 2):.6f}  {tab['duan_sum'][i]:.4f}  {tab['reid_ratio'][i]:.4f}")
