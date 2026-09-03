"""R18 — experiment proposal: CHSH with a Cooper-pair splitter, spin-resolved
detection and unequal arms (hypothesis H18, mechanism M1a unchanged,
experiment E18, role *proposal*).

This is an experiment design with numbers, not new theory.  Every closed
form is R10's (r10_cooper_pair_splitter.py is imported, not duplicated):
    S_obs = 2 sqrt2 p eta (2F_A - 1)(2F_B - 1),
    N_5sigma(S) = 25 (16 - S^2)/(S - 2)^2,
    verdict(S_meas, sigma, p, eta, F).
What this module adds is the evaluation of those forms with the pass-3
ledger numbers (constraints.md section E) for two platforms, the run-time
and decisiveness bands, and the unequal-arm (ordering) variant in physical
units.

Claims (numbered as in rows/R18_cps_proposal.md):
 1. Platform P1 (spin-polarised QD filters).  <C> = -P_1 P_2 E (L-E1a), so
    the per-side "fidelity" is F = (1 + P)/2 and S_obs = 2 sqrt2 p eta P1 P2.
    L-E1a: P_bar = 0.60 -> P1 P2 = 0.36 -> S_obs = 1.018 (p = eta = 1); the
    measured C = -0.37 read as the overall visibility -> S_obs = 1.047.
    L-E2c: C = -0.96 -> S_obs = 2.715 if the same visibility holds at all
    four CHSH settings (2.444 if that C were already efficiency-corrected
    and the > 90 % splitting efficiency applied on top).  Threshold: P1 P2 > 1/sqrt2 (symmetric P > 0.8409 at
    p eta = 1, > 0.9122 at eta = 0.85); L-E1b's 90 % visibility -> 2.546.
 2. Platform P2 (capture in gate-defined dots, single-shot readout).
    Readout F_r = 0.996 (L-E3b); Bell-state fidelity 0.9717 (L-E3a) ->
    Werner p_W = (4 F_s - 1)/3 = 0.9623.  Transfer fidelity F_t is a free
    parameter (spin-flip model per side, factor (2F_t - 1)):
    S_obs(F_t) = 2 sqrt2 p eta (2F_t-1)^2 (2F_r-1)^2 = 2.678 at F_t = 1;
    S_obs > 2 needs F_t > 0.9321 (flip model) or > 0.8642 (depolarising
    model, factor F_t per side).
 3. Statistics and run time.  N_5sigma = 422 (P1, S = 2.715), 480 (P2,
    S = 2.678); run time N/R for R = 1, 10, 100, 1000 s^-1.  The ledger has
    only DC currents: 100 pA = 6.2e8 e/s (L-E1b); a 2.4 us single-shot
    readout (L-E3b) caps a P2 cycle at 4.2e5 s^-1.
 4. Decisiveness bands (N = 1e4 coincidences, V_cal = 0.96): S <= 2 with
    V_cal > 0.7071 -> decisive negative; 2 < S < 2 + 5 sigma_S = 2.147 ->
    inconclusive; S >= 2.147 -> violation, "at the predicted level" if within
    2.715 +- 3 sigma_S = [2.627, 2.803].
 5. Unequal arms.  Drift 1e4 m/s, L_A = 2 um, L_B/L_A = 1, 2, 5: lab-time
    differences 0, 0.2, 0.8 ns (< 1e-3 of the > 1 us echo time L-C5a and
    the 2.4 us readout L-E3b); in frames A and B the pair separation is 0 at
    both filter events; the order (A first / B first) is the same in every
    frame; S is identical for all ratios (sequential projection in either
    order = the state's box).  Null-test sensitivity: with 1e4 coincidences
    per ratio a dependence |Delta S| >= 0.208 would show at 5 sigma.
 6. Verdict examples via r10.verdict.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import qm, frames
from mapping_spaces.entangled import r10_cooper_pair_splitter as r10

SQRT2 = np.sqrt(2.0)
E_CHARGE = 1.602176634e-19  # C

# ---- ledger values (constraints.md section E and earlier; not derived here)
LEDGER = {
    "P_bar_E1a": 0.60,        # mean filter polarisation, Bordoloi 2022
    "C_E1a": -0.37,           # measured spin cross-correlation, Bordoloi 2022
    "eta_2dot_E1a": 0.85,     # CPS fraction of the two-dot signal, Bordoloi 2022
    "vis_E1b": 0.90,          # combined CPS visibility, Wang 2022 (nanowire)
    "I_dc_E1b": 100e-12,      # A, currents ~100 pA, Wang 2022
    "C_E2c": -0.96,           # spin cross-correlation, Wang 2023 (2DEG)
    "eta_E2c": 0.90,          # combined efficiency > 90 %, Wang 2023
    "F_bell_E3a": 0.9717,     # Bell-state fidelity, Steinacker 2025
    "S_E3a": 2.731,           # co-located dot CHSH, Steinacker 2025
    "F_readout_E3b": 0.996,   # single-shot parity readout, Takeda 2024
    "t_readout_E3b": 2.4e-6,  # s, Takeda 2024
    "T_echo_C5a": 1e-6,       # s, echo > 1 us, Petta 2005 (lower bound)
    "T2star_C5a": 10e-9,      # s, Petta 2005
}


# ----------------------------------------------------- claim 1: platform P1
def fidelity_from_polarisation(P: float) -> float:
    """A filter of polarisation P passes the wrong spin with probability
    (1 - P)/2: the R10 per-side fidelity is F = (1 + P)/2, so 2F - 1 = P."""
    return 0.5 * (1.0 + P)


def s_obs_filters(p: float, eta: float, P1: float, P2: float) -> float:
    """S_obs for spin-polarised filters, through R10's closed form."""
    return r10.s_obs(p, eta, fidelity_from_polarisation(P1), fidelity_from_polarisation(P2))


def s_from_visibility(V: float) -> float:
    """S_obs when the measured spin correlation |C| is read as the overall
    visibility p eta P1 P2 (assumed equal at all four CHSH settings)."""
    return float(qm.TSIRELSON * V)


def p_min_symmetric(eta: float, p: float = 1.0) -> float:
    """Smallest symmetric polarisation with S_obs > 2: P^2 p eta > 1/sqrt2."""
    return float(np.sqrt(r10.VISIBILITY_THRESHOLD / (p * eta)))


def platform1() -> dict:
    L = LEDGER
    Pbar = L["P_bar_E1a"]
    return {
        "E1a_polarisation": s_obs_filters(1.0, 1.0, Pbar, Pbar),      # P1 P2 = 0.36
        "E1a_P1P2": Pbar * Pbar,
        "E1a_F_eff": fidelity_from_polarisation(Pbar),
        "E1a_correlation": s_from_visibility(abs(L["C_E1a"])),       # V = 0.37
        "E1b_visibility": s_from_visibility(L["vis_E1b"]),           # V = 0.90
        "E2c": s_from_visibility(abs(L["C_E2c"])),                   # V = 0.96
        "E2c_times_eta": s_from_visibility(abs(L["C_E2c"]) * L["eta_E2c"]),  # if C were efficiency-corrected
        "E2c_P_sym": float(np.sqrt(abs(L["C_E2c"]))),                # if p = eta = 1
        "E2c_F_eff": fidelity_from_polarisation(float(np.sqrt(abs(L["C_E2c"])))),
        "P_min_eta1": p_min_symmetric(1.0),
        "P_min_eta085": p_min_symmetric(L["eta_2dot_E1a"]),
        "F_min_r10": r10.f_min(1.0),  # the same threshold in R10's fidelity language
    }


# ----------------------------------------------------- claim 2: platform P2
def werner_p_from_fidelity(F_s: float) -> float:
    """Werner weight with singlet fidelity F_s = p + (1 - p)/4."""
    return (4.0 * F_s - 1.0) / 3.0


def transfer_factor(F_t: float, model: str = "flip") -> float:
    """Per-side correlator factor of a spin-preserving transfer of fidelity
    F_t: a spin flip with probability 1 - F_t gives 2F_t - 1 (R10's channel);
    full depolarisation with probability 1 - F_t gives F_t."""
    if model == "flip":
        return 2.0 * F_t - 1.0
    if model == "depol":
        return F_t
    raise ValueError(model)


def s_obs_dots(p: float, eta: float, F_t: float, F_r: float, model: str = "flip") -> float:
    """S_obs for capture-and-readout: transfer factor per side times R10's
    readout form.  For the flip model this is r10.s_obs with an effective
    per-side fidelity F_eff = (1 + (2F_t - 1)(2F_r - 1))/2."""
    f = transfer_factor(F_t, model)
    return float(r10.s_obs(p, eta, F_r, F_r) * f * f)


def ft_threshold(p: float, eta: float, F_r: float, model: str = "flip") -> float:
    """Smallest transfer fidelity with S_obs > 2."""
    need = np.sqrt(r10.VISIBILITY_THRESHOLD / (p * eta * (2 * F_r - 1) ** 2))  # per-side factor
    if model == "flip":
        return float(0.5 * (1.0 + need))
    return float(need)


def platform2(Fts=(1.0, 0.99, 0.98, 0.95, 0.93, 0.90), eta: float = 1.0) -> dict:
    L = LEDGER
    pW = werner_p_from_fidelity(L["F_bell_E3a"])
    Fr = L["F_readout_E3b"]
    return {
        "p_W": pW,
        "S_ledger_dots_V": r10.visibility_from_S(L["S_E3a"]),
        "table_flip": {Ft: s_obs_dots(pW, eta, Ft, Fr, "flip") for Ft in Fts},
        "table_depol": {Ft: s_obs_dots(pW, eta, Ft, Fr, "depol") for Ft in Fts},
        "S_direct_fidelity": s_obs_dots(L["F_bell_E3a"], eta, 1.0, Fr, "flip"),
        "Ft_min_flip": {e: ft_threshold(pW, e, Fr, "flip") for e in (1.0, 0.9, 0.85)},
        "Ft_min_depol": {e: ft_threshold(pW, e, Fr, "depol") for e in (1.0, 0.9, 0.85)},
    }


# -------------------------------------------- claim 3: statistics, run time
RATES = (1.0, 10.0, 100.0, 1000.0)


def run_time_table(S: float, rates=RATES) -> dict:
    """Coincidences for 5 sigma (R10 claim 4) and run time N/R in seconds."""
    N = r10.n_5sigma(S)
    return {"N": N, "t": {R: (N / R if N else np.inf) for R in rates}}


def electrons_per_second(I: float) -> float:
    return float(I / E_CHARGE)


def max_cycle_rate(t_readout: float) -> float:
    """Upper bound on a capture-and-readout coincidence rate: one readout
    per cycle."""
    return float(1.0 / t_readout)


# --------------------------------------------------- claim 4: decisiveness
def decisiveness_bands(V_cal: float, N: int) -> dict:
    """Which measured S counts as which outcome, given the independently
    calibrated visibility V_cal and N coincidences."""
    S_pred = s_from_visibility(V_cal)
    sig = r10.sigma_S(S_pred, N)
    return {
        "V_threshold": r10.VISIBILITY_THRESHOLD,
        "predicted_violation": V_cal > r10.VISIBILITY_THRESHOLD,
        "S_pred": S_pred,
        "sigma_S": sig,
        "S_5sigma": 2.0 + 5.0 * sig,                       # below: inconclusive
        "band_pred": (S_pred - 3.0 * sig, min(S_pred + 3.0 * sig, float(qm.TSIRELSON))),
    }


# ---------------------------------------------------- claim 5: unequal arms
def unequal_arms(v: float = 1e4, LA: float = 2e-6, ratios=(1.0, 2.0, 5.0), vB=None) -> dict:
    """Arms of lengths L_A and L_B = ratio * L_A at drift speed v (m/s),
    optionally a different speed vB on arm B, in R10's geometry (frames.py)."""
    vB = v if vB is None else vB
    out = {}
    for r in ratios:
        LB = r * LA
        g = r10.asymmetric_arms(v, vB, LA, LB)
        dt = g["yB"] - g["yA"]
        out[r] = {
            "yA": g["yA"], "yB": g["yB"], "dt": dt,
            "first": g["first"],
            "order_all_frames": r10.order_all_frames(g["a"], g["yA"], g["yB"]),
            "lab_sep_at_yA": g["sep_lab_at_yA"], "lab_sep_at_yB": g["sep_lab_at_yB"],
            "frame_A_sep": (g["sep_frame_A_at_yA"], g["sep_frame_A_at_yB"]),
            "frame_B_sep": (g["sep_frame_B_at_yA"], g["sep_frame_B_at_yB"]),
            "dt_over_T_echo": dt / LEDGER["T_echo_C5a"],
            "dt_over_t_readout": dt / LEDGER["t_readout_E3b"],
            "dt_over_T2star": dt / LEDGER["T2star_C5a"],
        }
    return out


def s_vs_ratio(p: float, ratios=(1.0, 2.0, 5.0)) -> dict:
    """Predicted S for each arm ratio: the sequential-projection box in the
    order the arms impose (A first for ratio > 1) equals the state's box, so
    S is the same number for every ratio (R10 claim 5)."""
    oi = r10.order_independence(p)
    out = {}
    for r in ratios:
        first = frames.lab_time_order(1.0, r)  # y_A = L_A/v, y_B = r L_A/v
        S = oi["S_AB"] if first in (frames.A, 0) else oi["S_BA"]
        out[r] = {"first": first, "S": S, "S_ref": oi["S_ref"], "max_box_diff": max(oi["diff_AB"], oi["diff_BA"])}
    return out


def null_test_sensitivity(S: float, N: int) -> dict:
    """Smallest |S(ratio) - S(1)| detectable with N coincidences per ratio."""
    sig_diff = SQRT2 * r10.sigma_S(S, N)
    return {"sigma_diff": sig_diff, "dS_5sigma": 5.0 * sig_diff, "dS_2sigma": 2.0 * sig_diff,
            "rel_5sigma": 5.0 * sig_diff / S}


# ------------------------------------------------------- claim 6: verdicts
def verdict_examples() -> dict:
    """R10's rule applied to the proposal's cases (symmetric F = (1+P)/2)."""
    P_2deg = float(np.sqrt(abs(LEDGER["C_E2c"])))
    F_2deg = fidelity_from_polarisation(P_2deg)
    F_nw = fidelity_from_polarisation(LEDGER["P_bar_E1a"])
    return {
        "2DEG_S2.7": r10.verdict(2.70, 0.03, 1.0, 1.0, F_2deg),
        "2DEG_S1.9": r10.verdict(1.90, 0.03, 1.0, 1.0, F_2deg),
        "2DEG_S2.1": r10.verdict(2.10, 0.03, 1.0, 1.0, F_2deg),
        "nanowire_S1.0": r10.verdict(1.02, 0.03, 1.0, 1.0, F_nw),
    }


if __name__ == "__main__":
    p1 = platform1()
    print("claim 1", p1)
    p2 = platform2()
    print("claim 2", p2)
    for name, S in (("P1 2DEG", p1["E2c"]), ("P2 Ft=1", p2["table_flip"][1.0]), ("P2 Ft=0.95", p2["table_flip"][0.95])):
        print("claim 3", name, run_time_table(S))
    print("claim 3", electrons_per_second(LEDGER["I_dc_E1b"]), max_cycle_rate(LEDGER["t_readout_E3b"]))
    print("claim 4", decisiveness_bands(abs(LEDGER["C_E2c"]), 10_000))
    print("claim 5", unequal_arms())
    print("claim 5 vB", unequal_arms(vB=2e4))
    print("claim 5 S", s_vs_ratio(1.0), s_vs_ratio(0.96))
    print("claim 5 null", null_test_sensitivity(p1["E2c"], 10_000))
    print("claim 6", verdict_examples())
