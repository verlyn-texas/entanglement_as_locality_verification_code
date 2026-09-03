"""R09 — projection-only collapse versus dynamical collapse models
(hypothesis H9, mechanism M7 = M1 contrasted with CSL / GRW / DP,
experiment E9, role U against the one empirically distinct rival).

The qD mechanism keeps collapse = projection (feedback 2).  Projection is a
state *update* conditioned on a measurement record; nothing happens to an
unmeasured particle's momentum distribution.  Dynamical collapse models
(CSL, GRW, Diósi–Penrose) instead add a stochastic localisation term that
acts on every particle all the time, and a Lindblad double commutator of the
mass density is what the statistics see.  Its three unavoidable side
effects — momentum diffusion, heating, spontaneous radiation — are exactly
zero in the qD model and finite in CSL/DP.  That is the only tabletop
regime in which the qD hypothesis differs from a named rival.

Formulae used (SI, mass-proportional CSL; lambda is the rate for one
nucleon-mass unit m_0 = 1 amu, r_C the localisation length):

* heating power of a point particle of mass m (state independent):
      P_CSL = dE/dt = 3 hbar^2 lambda m / (4 m_0^2 r_C^2)
  Carlesso et al., Nat. Phys. 18, 243 (2022), arXiv:2203.04231, Eq. (3);
  identical to Bassi et al., Rev. Mod. Phys. 85, 471 (2013), Eq. (224)
  (there written with M and m_N).
* momentum diffusion per Cartesian direction, D_p = d<p_x^2>/dt, follows
  from E = p^2/(2m) with three directions: dE/dt = 3 D_p / (2 m), so
      D_p = lambda hbar^2 m^2 / (2 m_0^2 r_C^2).
  In force-noise language D_p = hbar^2 eta, eta the CSL diffusion constant of
  the optomechanics literature (Nimmrichter, Hornberger, Hammerer, PRL 113,
  020405 (2014); Vinante et al., PRL 116, 090402 (2016)), whose general form
      eta = lambda r_C^3 / (pi^{3/2} m_0^2) Int d^3k e^{-k^2 r_C^2} k_x^2 |rho~(k)|^2
  reduces to the point value for |rho~|^2 = m^2 (checked in ``sphere_form_factor``).
* CSL position spread of a free particle (3-D):
      <x^2>_t = <x^2>_t^QM + lambda hbar^2 t^3 / (2 m_0^2 r_C^2)
  Carlesso et al. Eq. (4); equals Sum_i D_p t^3 / (3 m^2) with D_p above.
* coherence between two positions d apart decays at
      Gamma(d) = lambda (m/m_0)^2 (1 - exp(-d^2 / 4 r_C^2))
  (Bassi et al. 2013, the standard mass-proportional result); the discrete
  two-site model below uses the same Gaussian kernel and reproduces it.
* Diósi–Penrose heating of a point particle with smearing R_0 (derived here
  from the DP master equation, prefactor G/(2 hbar), with a normalised Gaussian
  mass profile of variance R_0^2 per axis; the review quotes "of the order of
  1e-20 erg/s" for a nucleon at R_0 = 1e-15 m, which this coefficient reproduces):
      dE/dt = G hbar m / (4 sqrt(pi) R_0^3).
  Verified against Nimmrichter, Hornberger, Hammerer, PRL 113, 020405 (2014),
  Eq. (11) + supplement Eq. (S4): D_DP = G hbar/(2 pi^2) Int d^3k (k_x^2/k^2) |rho~(k)|^2
  = G hbar m^2/(6 sqrt(pi) R_0^3) per axis for |rho~|^2 = m^2 e^{-k^2 R_0^2}
  (test_claim6).  Donadi et al. 2021 (Nat. Phys. 17, 74) use Penrose's 4 pi G/hbar
  prefactor, 8 pi times this, with the same Gaussian; their R_0 bounds are in
  that normalisation.

Claims (numbered as in rows/R09_vs_collapse_models.md):
 1. The qD model's diffusion, heating and radiation functions are identically
    zero for every (lambda, r_C, m): there is no stochastic term to feed them.
 2. CSL formulae: D_p and dE/dt are consistent (dE/dt = 3 D_p / 2m), give the
    tabulated numbers for an electron, a nucleon and a 1e-12 kg mass, and the
    homogeneous-sphere form factor tends to 1 for R << r_C and to 6 (r_C/R)^4
    for R >> r_C, so a 1e-12 kg sphere sees ~1e-5 of the point-particle
    diffusion (the multilayer of L-C2a is designed to beat this suppression).
 3. Electron table at r_C = 1e-7 m for lambda = 1e-16, 1e-10, 1e-8 s^-1: D_p,
    dE/dt in eV/s and the time for the momentum spread to reach 1e-3 of
    p_0 = m_e * 1e4 m/s (R07's reference momentum): 5e18, 5e12, 5e10 s.
 4. Spin factorisation: CSL acts on the mass density (positions) only, so for
    a spin (x) position product state the reduced spin state — hence CHSH —
    is unchanged by any amount of position decoherence; a spin–position
    singlet whose spin is *stored in a path superposition* while CSL acts
    (Stern-Gerlach split, dephase, recombine) is decohered at the sum of the
    two single-particle rates Gamma(d), which for electrons at
    lambda = 2e-10 s^-1 is 1.2e-16 s^-1 (5e-9 per year).
 5. Exclusion status of the three lambda values against L-C2a/c/d, and the
    target: excluding GRW at r_C = 1e-7 m needs lambda < 1e-16 s^-1, a factor
    5.2e3 beyond the review's X-ray bound (L-C2d) and 1.7e3 beyond MAJORANA's
    quasi-free-electron bound (L-C2c).
 6. Diósi–Penrose: the parameter-free choice R_0 ~ 1e-15 m gives a nucleon
    heating ~1.7e-20 erg/s (excluded, L-C2b/c); at the surviving
    R_0 > 2.54e-10 m the rate is 1.6e-16 of that; the qD prediction is zero.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import qm

# ------------------------------------------------------------------ constants
HBAR = 1.054571817e-34      # J s
AMU = 1.66053907e-27        # kg  (m_0 of mass-proportional CSL)
M_E = 9.1093837e-31         # kg
M_NUCLEON = 1.6726e-27      # kg  (proton mass, used as "a nucleon")
G_NEWTON = 6.67430e-11      # m^3 kg^-1 s^-2
EV = 1.602176634e-19        # J
K_B = 1.380649e-23          # J/K
YEAR = 3.15576e7            # s

R_C_REF = 1e-7              # m, the reference localisation length of the ledger

# Ledger values (registers/constraints.md, verified 2026-08-25)
LAMBDA_GRW = 1e-16                 # s^-1, GRW proposal (L-C2d: allowed)
LAMBDA_CANTILEVER = 2.0e-10        # s^-1, L-C2a (via review L-C2d), r_C = 1e-7 m
LAMBDA_XRAY_REVIEW = 5.2e-13       # s^-1, L-C2d (X-ray, r_C = 1e-7 m)
LAMBDA_XRAY_MAJORANA = 1.7e-13     # s^-1, L-C2c (quasi-free electrons, white noise)
LAMBDA_XRAY_NUCLEAR = 4.9e-15      # s^-1, L-C2c (with coherent nuclear emission)
LAMBDA_LISA = 3.8e-9               # s^-1, L-C2d
LAMBDA_COLD_ATOMS = 5.1e-8         # s^-1, L-C2d
LAMBDA_ADLER = 1e-8                # s^-1, the "Adler value" used in the table (L-C2d: 10^{-8±2}, excluded)
R0_DP_DONADI = 0.54e-10            # m, L-C2b (95% CL lower bound)
R0_DP_MAJORANA = 2.54e-10          # m, L-C2c
R0_DP_NUCLEAR = 1e-15              # m, the parameter-free (nuclear-size) DP choice

P0_R07 = M_E * 1e4                 # kg m/s: R07's reference electron momentum (1e4 m/s)


# ------------------------------------------------- claim 1: the qD model (M7)
def qd_momentum_diffusion(lam, r_c, m) -> float:
    """d<p_x^2>/dt of an unmeasured particle in the qD model with collapse =
    projection.  There is no stochastic localisation term, so the answer is
    identically zero — for every lambda, r_C and m one might try to attach."""
    return 0.0 * float(np.asarray(lam)) * float(np.asarray(r_c)) * float(np.asarray(m))


def qd_heating_rate(lam, r_c, m) -> float:
    return 0.0 * qd_momentum_diffusion(lam, r_c, m)


def qd_radiation_rate(lam, r_c, m) -> float:
    """Spontaneous photon emission rate: zero (no accelerating noise)."""
    return 0.0 * qd_momentum_diffusion(lam, r_c, m)


# ------------------------------------------------------- claim 2: CSL formulae
def csl_momentum_diffusion(lam, r_c, m, m0: float = AMU):
    """D_p = d<p_x^2>/dt per Cartesian direction, point particle [kg^2 m^2 s^-3 = N^2 s]."""
    return lam * HBAR**2 * m**2 / (2.0 * m0**2 * r_c**2)


def csl_heating_rate(lam, r_c, m, m0: float = AMU):
    """dE/dt = 3 hbar^2 lambda m / (4 m_0^2 r_C^2)  [W]  (Carlesso Eq. 3, Bassi Eq. 224)."""
    return 3.0 * HBAR**2 * lam * m / (4.0 * m0**2 * r_c**2)


def csl_position_spread(lam, r_c, t, m0: float = AMU):
    """CSL-induced addition to <x^2> (3-D) after free evolution for time t
    (Carlesso Eq. 4): lambda hbar^2 t^3 / (2 m_0^2 r_C^2).  Mass independent."""
    return lam * HBAR**2 * t**3 / (2.0 * m0**2 * r_c**2)


def csl_decoherence_rate(lam, r_c, m, d, m0: float = AMU):
    """Decay rate of the coherence between two positions d apart."""
    return lam * (m / m0) ** 2 * (1.0 - np.exp(-(d**2) / (4.0 * r_c**2)))


def sphere_form_factor(R: float, r_c: float, n: int = 40001, u_max: float = 8.0) -> float:
    """eta_sphere / eta_point for a homogeneous sphere of radius R:
    (8 / 3 sqrt(pi)) Int_0^inf u^4 e^{-u^2} F(u R / r_C)^2 du,
    F(x) = 3 (sin x - x cos x) / x^3 the sphere's normalised form factor.
    Limits: 1 for R << r_C, 6 (r_C / R)^4 for R >> r_C."""
    u = np.linspace(0.0, u_max, n)
    x = u * R / r_c
    F = np.ones_like(x)
    nz = x > 1e-6
    F[nz] = 3.0 * (np.sin(x[nz]) - x[nz] * np.cos(x[nz])) / x[nz] ** 3
    integrand = u**4 * np.exp(-(u**2)) * F**2
    return float(8.0 / (3.0 * np.sqrt(np.pi)) * np.trapezoid(integrand, u))


def sphere_radius(m: float, density: float) -> float:
    return float((3.0 * m / (4.0 * np.pi * density)) ** (1.0 / 3.0))


def csl_momentum_diffusion_sphere(lam, r_c, m, R, m0: float = AMU) -> float:
    """D_p of a homogeneous sphere (mass-density coupling), per direction."""
    return float(csl_momentum_diffusion(lam, r_c, m, m0) * sphere_form_factor(R, r_c))


def csl_examples(lam: float = LAMBDA_CANTILEVER, r_c: float = R_C_REF,
                 m_test: float = 1e-12, density: float = 7.8e3) -> dict:
    """The three bodies of claim 2 at one (lambda, r_C)."""
    R = sphere_radius(m_test, density)
    out = {}
    for name, m in (("electron", M_E), ("nucleon", M_NUCLEON), ("test_mass_point", m_test)):
        out[name] = {"D_p": float(csl_momentum_diffusion(lam, r_c, m)),
                     "dEdt_W": float(csl_heating_rate(lam, r_c, m)),
                     "dEdt_eV_s": float(csl_heating_rate(lam, r_c, m) / EV)}
    ff = sphere_form_factor(R, r_c)
    out["test_mass_sphere"] = {"radius_m": R, "R_over_rC": R / r_c, "form_factor": ff,
                               "asymptote_6_rC4_R4": 6.0 * (r_c / R) ** 4,
                               "D_p": out["test_mass_point"]["D_p"] * ff}
    return out


# ------------------------------------------- claim 3: the free-electron table
def electron_table(lams=(LAMBDA_GRW, LAMBDA_CANTILEVER / 2, LAMBDA_ADLER), r_c: float = R_C_REF,
                   p0: float = P0_R07, fraction: float = 1e-3) -> list[dict]:
    """For each lambda: D_p, heating in eV/s, and the time for sqrt(<Delta p_x^2>)
    to reach fraction * p0 (Delta p^2 = D_p t)."""
    rows = []
    for lam in lams:
        Dp = float(csl_momentum_diffusion(lam, r_c, M_E))
        dE = float(csl_heating_rate(lam, r_c, M_E))
        t = (fraction * p0) ** 2 / Dp
        rows.append({"lambda": lam, "D_p": Dp, "dEdt_eV_s": dE / EV, "dEdt_K_s": dE / (1.5 * K_B),
                     "t_to_fraction_s": t, "t_to_fraction_yr": t / YEAR})
    return rows


# ------------------------------------ claim 4: spin (x) position factorisation
# Hilbert space: spin_A (x) spin_B (x) pos_A (x) pos_B, each factor 2-dim; the
# position factor has two sites per particle.  The CSL generator is the
# mass-proportional double commutator of the site mass densities with the
# Gaussian kernel K(r) = lambda (m/m_0)^2 exp(-r^2 / 4 r_C^2) / ... written so
# that a single particle superposed over d loses coherence at Gamma(d) above.
SITES_DEFAULT = {"A": (-1.0e-3, -1.0e-3 + 1e-6), "B": (1.0e-3, 1.0e-3 + 1e-6)}  # metres


def _position_configs(sites):
    """The four position basis states (sA, sB) -> list of site coordinates
    occupied, as arrays of x for particle A and B."""
    xa, xb = sites["A"], sites["B"]
    return [(xa[i], xb[j]) for i in range(2) for j in range(2)]


def csl_dephasing_matrix(lam, r_c, m, sites=SITES_DEFAULT, m0: float = AMU) -> np.ndarray:
    """Gamma[c, c'] = decay rate of the coherence between position
    configurations c and c' (each a pair of occupied sites), from
    (lambda / 2 m_0^2) Sum_{s,s'} K(x_s - x_s') Dn_s Dn_s', with
    K(r) = exp(-r^2 / 4 r_C^2) and Dn_s the mass difference at site s."""
    configs = _position_configs(sites)
    all_sites = sorted({x for pair in configs for x in pair})
    n = np.zeros((4, len(all_sites)))
    for c, pair in enumerate(configs):
        for x in pair:
            n[c, all_sites.index(x)] += m
    xs = np.array(all_sites)
    K = np.exp(-((xs[:, None] - xs[None, :]) ** 2) / (4.0 * r_c**2))
    G = np.zeros((4, 4))
    for c in range(4):
        for cp in range(4):
            dn = n[c] - n[cp]
            G[c, cp] = lam / (2.0 * m0**2) * dn @ K @ dn
    return G


def position_decoherence(rho16: np.ndarray, gamma_t: np.ndarray) -> np.ndarray:
    """Apply exp(-Gamma t) dephasing to the position factor (last 4-dim
    factor of spin_A x spin_B x pos_A x pos_B): rho_{(s,c),(s',c')} *=
    exp(-gamma_t[c, c']).  Acts as the identity on the spin factor."""
    r = rho16.reshape(4, 4, 4, 4).copy()   # (spin, pos, spin', pos')
    r *= np.exp(-gamma_t)[None, :, None, :]
    return r.reshape(16, 16)


def reduced_spin(rho16: np.ndarray) -> np.ndarray:
    r = rho16.reshape(4, 4, 4, 4)
    return np.trace(r, axis1=1, axis2=3)


def product_singlet_state() -> np.ndarray:
    """Singlet (x) (each particle in an equal superposition of its two sites)."""
    pos = qm.ket(1, 1, 1, 1)
    return qm.dm(np.kron(qm.singlet(), pos))


def spin_to_path_unitary() -> np.ndarray:
    """V = CNOT_A (x) CNOT_B with the spin as control: a spin-down particle is
    moved to its other site.  This is what a Stern-Gerlach stage does — it
    writes the spin into the path; V^dagger recombines the paths."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    V = np.zeros((16, 16), dtype=complex)
    for sA in range(2):
        for sB in range(2):
            P = qm.kron(qm.dm(np.eye(2)[sA]), qm.dm(np.eye(2)[sB]))
            V += qm.kron(P, X if sA else qm.I2, X if sB else qm.I2)
    return V


def interferometric_singlet_state() -> np.ndarray:
    """Singlet (x) |L>_A |L>_B before the spin-to-path stage."""
    return qm.dm(np.kron(qm.singlet(), qm.ket(1, 0, 0, 0)))


def factorisation_check(lam: float = LAMBDA_CANTILEVER, r_c: float = R_C_REF, m: float = M_E,
                        t: float = YEAR, exaggerate: float = 1.0) -> dict:
    """CHSH_max of the reduced spin state before and after CSL position
    dephasing, for (i) the product singlet (spin never written into path)
    and (ii) the interferometric sequence V -> dephase -> V^dagger (spin
    stored in a path superposition while CSL acts).  ``exaggerate``
    multiplies the physical rates to show the fully decohered limit."""
    G = csl_dephasing_matrix(lam, r_c, m) * t * exaggerate
    V = spin_to_path_unitary()
    out = {"gamma_t_max": float(G.max())}
    rho = product_singlet_state()
    after = position_decoherence(rho, G)
    out["product"] = {"S_before": qm.chsh_max(reduced_spin(rho)), "S_after": qm.chsh_max(reduced_spin(after)),
                      "concurrence_after": qm.concurrence(reduced_spin(after))}
    rho = interferometric_singlet_state()
    stored = V @ rho @ V.conj().T
    recombined = V.conj().T @ position_decoherence(stored, G) @ V
    out["interferometric"] = {"S_before": qm.chsh_max(reduced_spin(V.conj().T @ stored @ V)),
                              "S_after": qm.chsh_max(reduced_spin(recombined)),
                              "concurrence_after": qm.concurrence(reduced_spin(recombined))}
    # rate for the coherence between (A at L, B at R) and (A at R, B at L) = the
    # two configurations the stored singlet occupies: config index 1 vs 2
    out["two_particle_rate_s"] = float(G[1, 2] / (t * exaggerate))
    out["single_particle_rate_s"] = float(csl_decoherence_rate(lam, r_c, m, 1e-6))
    return out


# --------------------------------------------- claim 5: exclusion and target
BOUNDS_AT_RC_REF = {
    "L-C2c X-ray (MAJORANA, quasi-free e-)": LAMBDA_XRAY_MAJORANA,
    "L-C2d X-ray (review)": LAMBDA_XRAY_REVIEW,
    "L-C2a/L-C2d cantilever": LAMBDA_CANTILEVER,
    "L-C2d LISA Pathfinder": LAMBDA_LISA,
    "L-C2d cold atoms": LAMBDA_COLD_ATOMS,
}


def exclusion_status(lam: float, bounds: dict = BOUNDS_AT_RC_REF) -> dict:
    """Which verified bounds (at r_C = 1e-7 m) exclude this lambda."""
    excluded_by = [k for k, b in bounds.items() if lam >= b]
    return {"lambda": lam, "excluded": bool(excluded_by), "excluded_by": excluded_by,
            "allowed_by_cantilever_alone": lam < LAMBDA_CANTILEVER}


def grw_target() -> dict:
    """Bound needed to exclude the GRW value, and the improvement over the
    present best bounds."""
    return {"target_lambda": LAMBDA_GRW,
            "factor_vs_xray_review": LAMBDA_XRAY_REVIEW / LAMBDA_GRW,
            "factor_vs_xray_majorana": LAMBDA_XRAY_MAJORANA / LAMBDA_GRW,
            "factor_vs_xray_nuclear": LAMBDA_XRAY_NUCLEAR / LAMBDA_GRW,
            "factor_vs_cantilever": LAMBDA_CANTILEVER / LAMBDA_GRW}


# ------------------------------------------------------ claim 6: Diósi–Penrose
def dp_momentum_diffusion_total(m, R0):
    """d<p^2>/dt summed over the three directions, point particle smeared
    over a Gaussian of variance R_0^2 per axis (rho ~ exp(-r^2 / 2 R_0^2)),
    DP prefactor G/(2 hbar): 4 pi G hbar Int rho^2 d^3x = G hbar m^2 / (2 sqrt(pi) R_0^3)."""
    return G_NEWTON * HBAR * m**2 / (2.0 * np.sqrt(np.pi) * R0**3)


def dp_heating_rate(m, R0):
    """dE/dt = G hbar m / (4 sqrt(pi) R_0^3)  [W]."""
    return dp_momentum_diffusion_total(m, R0) / (2.0 * m)


def dp_examples() -> dict:
    p = dp_heating_rate(M_NUCLEON, R0_DP_NUCLEAR)
    return {"nucleon_R0_1e-15_W": float(p), "nucleon_R0_1e-15_erg_s": float(p * 1e7),
            "nucleon_R0_1e-15_K_s": float(p / (1.5 * K_B)),
            "nucleon_R0_2.54e-10_W": float(dp_heating_rate(M_NUCLEON, R0_DP_MAJORANA)),
            "suppression_at_2.54e-10": float((R0_DP_NUCLEAR / R0_DP_MAJORANA) ** 3),
            "qd_prediction_W": qd_heating_rate(0.0, R0_DP_NUCLEAR, M_NUCLEON)}


if __name__ == "__main__":
    print("claim 1  qD diffusion / heating / radiation:",
          qd_momentum_diffusion(1e-8, 1e-7, M_E), qd_heating_rate(1e-8, 1e-7, M_E), qd_radiation_rate(1e-8, 1e-7, M_E))
    print("claim 2  CSL examples at lambda = 2e-10, r_C = 1e-7 m:")
    for k, v in csl_examples().items():
        print("   ", k, v)
    print("claim 3  electron table at r_C = 1e-7 m:")
    for row in electron_table():
        print("   ", {k: f"{v:.3g}" for k, v in row.items()})
    print("claim 4  factorisation (physical rates, 1 yr):", factorisation_check())
    print("claim 4  factorisation (exaggerated x1e17):", factorisation_check(exaggerate=1e17))
    print("claim 5  exclusion:")
    for lam in (LAMBDA_GRW, 1e-10, LAMBDA_ADLER):
        print("   ", exclusion_status(lam))
    print("         target:", grw_target())
    print("claim 6  DP:", dp_examples())
