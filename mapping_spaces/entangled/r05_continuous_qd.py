"""R05 — mechanism M4: continuous qD tied to the degree of entanglement.

The pair sits at z = -C (A) and z = +C (B), C = Wootters concurrence of its
spin state.  The frame maps are the z-dependent shears of
``solutions/transformation.md`` with shear potentials

    sigma_1 = s(z)/a,   sigma_2 = -s(z)/a,   sigma_3 = t(z)/a,

pinned by the table only at z in {-1, 0, 1}: s(+-1) = 0, s(0) = 1 (even),
t(0) = 0, t(+-1) = +-1 (odd).  The quadratic interpolant of transformation.md
is s = 1 - z^2, t = z.

"Locality fraction" = the fraction of the lab's separation rate that is
removed in the particles' own frames.  Two conventions for where a pair at
height +-C sits, and the interpolant, all enter; this module computes every
variant so the document can show which numbers are convention-free.

Claims (numbered as in rows/R05_continuous_qd.md):
 1. At rest in their own frames (frames.separation_in_frame, quadratic
    potentials) the separation rate in frame A or B is (1 - C^2) 2/a:
    fraction C^2 = frames.locality_fraction(C), C in {0, .25, .5, .75, 1}.
 2. Under that convention the lab sees the rate 2(1 + C - C^2)/a, i.e. 2.5/a
    at C = 1/2: a is then not the observed separation speed.
 3. Pinning the lab tracks to the observed x = -+ y/a (general paths) gives
    the particle-frame rate 2(1 - t(C))/a for EVERY even s: fraction C for
    t = z, independent of the sigma_{1,2} interpolant; t = z^3 gives C^3.
    Re-pinning the table at +-C gives fraction 1 for all C > 0.
 4. Werner states: C = max(0, (3p-1)/2), S_max = 2 sqrt2 p.
 5. Non-maximal pure states cos t|01> - sin t|10>: C = sin 2t,
    S_max = 2 sqrt(1 + C^2).
 6. C is not a function of S: equal C with different S, and equal S with
    different C, are both exhibited.
 7. Ledger illustration under a Werner assumption p = S/(2 sqrt2).
 8. Every spin observable of M4 equals that of QM/M1 (same box, no
    signalling) for random states; the only new number is ell(C).
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import frames, qm

C_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

# ------------------------------------------------------------- interpolants
# even parts s(z): s(0) = 1, s(+-1) = 0
S_INTERP = {
    "quadratic": lambda z: 1.0 - z * z,               # transformation.md
    "piecewise_linear": lambda z: 1.0 - np.abs(z),
    "quartic": lambda z: (1.0 - z * z) ** 2,
    "cosine": lambda z: np.cos(np.pi * z / 2.0),
}
# odd parts t(z): t(0) = 0, t(+-1) = +-1
T_INTERP = {
    "linear": lambda z: z,                            # transformation.md
    "cubic": lambda z: z ** 3,
}


def sigma(k: int, a: float, s_name: str = "quadratic", t_name: str = "linear"):
    """Shear potential of frame k for the named interpolants."""
    s, t = S_INTERP[s_name], T_INTERP[t_name]
    if k == frames.A:
        return lambda z: s(np.asarray(z, float)) / a
    if k == frames.B:
        return lambda z: -s(np.asarray(z, float)) / a
    return lambda z: t(np.asarray(z, float)) / a


def beta_general(k: int, l: int, a: float, z, s_name="quadratic", t_name="linear"):
    return sigma(l, a, s_name, t_name)(z) - sigma(k, a, s_name, t_name)(z)


# ----------------------------------------------------- claims 1-3: geometry
def rates_at_rest(C: float, a: float = 2.0, s_name="quadratic", t_name="linear") -> dict:
    """Separation rate x_B - x_A per unit y in frames A, B and the lab when
    the pair at z = -+C is AT REST in its own frames (p_A = p_B = 0)."""
    out = {}
    for k, name in ((frames.A, "A"), (frames.B, "B"), (frames.LAB, "lab")):
        xA = beta_general(frames.A, k, a, -C, s_name, t_name)
        xB = beta_general(frames.B, k, a, +C, s_name, t_name)
        out[name] = float(xB - xA)
    return out


def rates_pinned(C: float, a: float = 2.0, s_name="quadratic", t_name="linear") -> dict:
    """Same, but the LAB tracks are pinned to the observed x = -+ y/a (the
    pair follows general paths p_i in its own frame, transformation.md S4)."""
    # own-frame paths that put the lab tracks at -+ y/a
    pA = -1.0 / a - beta_general(frames.A, frames.LAB, a, -C, s_name, t_name)
    pB = +1.0 / a - beta_general(frames.B, frames.LAB, a, +C, s_name, t_name)
    out = {"pA": float(pA), "pB": float(pB)}
    for k, name in ((frames.A, "A"), (frames.B, "B"), (frames.LAB, "lab")):
        xA = pA + beta_general(frames.A, k, a, -C, s_name, t_name)
        xB = pB + beta_general(frames.B, k, a, +C, s_name, t_name)
        out[name] = float(xB - xA)
    return out


def locality_fraction_table(Cs=C_GRID, a: float = 2.0) -> np.ndarray:
    """Columns: C, rate in B (at rest, quadratic), lab rate (at rest),
    frames.locality_fraction(C) = 1 - rate_B/(2/a), rate in B (pinned),
    pinned fraction 1 - rate_B_pinned/(2/a)."""
    rows = []
    for C in Cs:
        r0 = rates_at_rest(C, a)
        r1 = rates_pinned(C, a)
        sep_frames = frames.separation_in_frame(frames.B, a, np.array([1.0]), zA=-C, zB=C)[0]
        assert abs(sep_frames - r0["B"]) < 1e-12  # frames.py agrees with the general code
        rows.append([C, r0["B"], r0["lab"], frames.locality_fraction(C),
                     1 - r0["B"] / (2 / a), r1["B"], 1 - r1["B"] / (2 / a)])
    return np.array(rows)


def interpolant_table(Cs=C_GRID, a: float = 2.0) -> dict:
    """Locality fraction (relative to 2/a) for every interpolant, under both
    conventions.  Keys: (s_name, t_name, convention)."""
    out = {}
    for s_name in S_INTERP:
        for t_name in T_INTERP:
            at_rest = np.array([1 - rates_at_rest(C, a, s_name, t_name)["B"] / (2 / a) for C in Cs])
            pinned = np.array([1 - rates_pinned(C, a, s_name, t_name)["B"] / (2 / a) for C in Cs])
            out[(s_name, t_name, "at_rest")] = at_rest
            out[(s_name, t_name, "pinned")] = pinned
    return out


def repinned_fraction(C: float) -> float:
    """If the table of transformation.md is re-pinned at z = -+C (so that
    sigma_1(-+C) = sigma_2(-+C) = 0), beta_12(-+C) = 0 and the pair is fully
    co-located: fraction 1 for every C > 0, 0 at C = 0."""
    return 1.0 if C > 0 else 0.0


# ---------------------------------------------------- claims 4-6: states
def werner_table(ps=(1 / 3, 0.5, 1 / np.sqrt(2), 0.8, 0.9, 1.0)) -> np.ndarray:
    """Columns: p, C (numerical), (3p-1)/2 clipped, C^2, S_max (numerical), 2 sqrt2 p."""
    rows = []
    for p in ps:
        rho = qm.werner(p)
        C = qm.concurrence(rho)
        rows.append([p, C, max(0.0, (3 * p - 1) / 2), C * C, qm.chsh_max(rho), qm.TSIRELSON * p])
    return np.array(rows)


def nonmax_table(thetas=(0.0, np.pi / 16, np.pi / 8, 3 * np.pi / 16, np.pi / 4)) -> np.ndarray:
    """Columns: theta, C (numerical), sin 2theta, S_max (numerical), 2 sqrt(1 + sin^2 2theta)."""
    rows = []
    for th in thetas:
        psi = qm.nonmax_entangled(th)
        C = qm.concurrence(psi)
        rows.append([th, C, np.sin(2 * th), qm.chsh_max(psi), 2 * np.sqrt(1 + np.sin(2 * th) ** 2)])
    return np.array(rows)


def werner_threshold_C() -> float:
    """Concurrence at which a Werner state first violates CHSH (p = 1/sqrt2)."""
    return (3 / np.sqrt(2) - 1) / 2


def same_C_different_S(C: float = 0.7) -> dict:
    """A Werner state and a pure state with the same concurrence."""
    p = (2 * C + 1) / 3
    th = np.arcsin(C) / 2
    w, psi = qm.werner(p), qm.nonmax_entangled(th)
    return {"p": p, "theta": th,
            "C_werner": qm.concurrence(w), "C_pure": qm.concurrence(psi),
            "S_werner": qm.chsh_max(w), "S_pure": qm.chsh_max(psi)}


def same_S_different_C(p: float = 0.8) -> dict:
    """A Werner state and a pure state with the same CHSH maximum."""
    S = qm.TSIRELSON * p
    sin2t = np.sqrt(S * S / 4 - 1)
    th = np.arcsin(sin2t) / 2
    w, psi = qm.werner(p), qm.nonmax_entangled(th)
    return {"S": S, "S_werner": qm.chsh_max(w), "S_pure": qm.chsh_max(psi),
            "C_werner": qm.concurrence(w), "C_pure": qm.concurrence(psi)}


# ------------------------------------------------ claim 7: ledger mapping
def werner_from_S(S: float, dS: float = 0.0) -> dict:
    """Werner assumption p = S/(2 sqrt2): C = (3p-1)/2, ell_quadratic = C^2,
    ell_pinned = C, with linearly propagated 1-sigma errors.  Also the
    pure-state alternative C = sqrt(S^2/4 - 1)."""
    p = S / qm.TSIRELSON
    dp = dS / qm.TSIRELSON
    C = max(0.0, (3 * p - 1) / 2)
    dC = 1.5 * dp
    C_pure = float(np.sqrt(max(0.0, S * S / 4 - 1)))
    dC_pure = (S / 4) / C_pure * dS if C_pure > 0 else float("nan")
    return {"S": S, "p": p, "dp": dp, "C": C, "dC": dC, "C2": C * C, "dC2": 2 * C * dC,
            "F": (1 + 3 * p) / 4, "C_pure": C_pure, "dC_pure": dC_pure}


LEDGER_S = {
    "L-A6d 0.1 K (verified)": (2.731, 0.088),
    "L-A6d 1.1 K (NOT in ledger; caller's brief)": (2.101, 0.064),
    "L-A3 ions (verified)": (2.25, 0.03),
    "L-A4 photons (verified)": (2.82759, 0.00051),
}


def ledger_illustration() -> dict:
    return {k: werner_from_S(S, dS) for k, (S, dS) in LEDGER_S.items()}


# ------------------------------------------ claim 8: no new observable
def random_states(n: int = 200, seed: int = 5) -> list:
    """Random two-qubit density matrices (Ginibre) and random pure states."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        G = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho = G @ G.conj().T
        out.append(rho / np.trace(rho).real)
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        out.append(qm.dm(v / np.linalg.norm(v)))
    return out


def m4_box(state, settings_A, settings_B) -> np.ndarray:
    """M4's prediction for the box P(a,b|x,y): the spin state is untouched by
    the frame maps (they move x only), so it is QM's box.  The mechanism's
    only addition is the height z = -+C(rho)."""
    return qm.box_from_state(state, settings_A, settings_B)


def m4_vs_qm(n: int = 200, seed: int = 5) -> dict:
    a, ap, b, bp = qm.chsh_optimal_settings()
    max_diff, max_sig, Cs, Ss = 0.0, 0.0, [], []
    for rho in random_states(n, seed):
        box = m4_box(rho, (a, ap), (b, bp))
        max_diff = max(max_diff, float(np.abs(box - qm.box_from_state(rho, (a, ap), (b, bp))).max()))
        max_sig = max(max_sig, qm.signalling(box))
        Cs.append(qm.concurrence(rho))
        Ss.append(qm.chsh_max(rho))
    return {"max_box_difference": max_diff, "max_signalling": max_sig,
            "C": np.array(Cs), "S": np.array(Ss)}


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("claims 1-3: C | rate_B at rest | lab rate at rest | C^2 | 1-rate_B/(2/a) | rate_B pinned | pinned fraction")
    print(locality_fraction_table())
    for key, val in interpolant_table().items():
        print(f"  {key}: {val}")
    print("claim 4 (Werner): p, C, (3p-1)/2, C^2, S, 2sqrt2 p")
    print(werner_table())
    print("claim 5 (pure): theta, C, sin2theta, S, 2sqrt(1+sin^2)")
    print(nonmax_table())
    print("claim 6:", werner_threshold_C(), same_C_different_S(), same_S_different_C())
    print("claim 7:")
    for k, v in ledger_illustration().items():
        print(f"  {k}: p={v['p']:.3f}({v['dp']:.3f}) C={v['C']:.3f}({v['dC']:.3f}) "
              f"C^2={v['C2']:.3f}({v['dC2']:.3f}) F={v['F']:.3f} C_pure={v['C_pure']:.3f}({v['dC_pure']:.3f})")
    r = m4_vs_qm()
    print("claim 8:", r["max_box_difference"], r["max_signalling"], r["C"].min(), r["C"].max())
