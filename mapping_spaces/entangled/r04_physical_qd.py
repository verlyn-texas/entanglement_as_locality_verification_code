"""R04 -- mechanism M3: qD as a *physical* compact dimension (Kaluza-Klein).

The shear model of ``transformation.md`` uses a dimensionless height z with
the pair at z = -1, +1 and the lab at z = 0.  If qD is a physical dimension
it needs a length scale L (the lab-to-particle displacement in qD), a
physical coordinate Z = L z, and -- if the dimension is compact with
circumference 2 pi R -- the pair, being placed at Z = -L and Z = +L on the
same circle, needs 2 L <= pi R.

Two sub-variants:
* free motion  -- the electron is a field on the circle: a tower of
  excitations (Kaluza-Klein modes) and a wave packet that spreads in qD;
* placement    -- the electron is *put* at Z = +-L by the entangling event
  and has no dynamics in qD: no tower, no spreading.

Claims (numbered as in rows/R04_physical_qd.md):
 1. Z = L z reproduces every spacetime prediction of the shear model for
    every L > 0 (L drops out of all lab observables); compactness needs
    L <= pi R / 2.
 2. Massless Kaluza-Klein tower E_n = n hbar c / R: E_1 in eV over
    R = 1e-12 .. 1e-4 m, and the radii at which E_1 crosses 1 eV, kT and 1 keV.
 3. Non-relativistic ring E_n = n^2 hbar^2 / (2 m_e R^2): same table and
    thresholds; the exact massive KK excitation agrees with the ring for
    R >> hbar/(2 m_e c) = 1.93e-13 m.
 4. Free motion delocalises a packet placed at Z = +-L in tau <= m_e R^2/hbar
    (7.8 us at R = 30 um), far shorter than the stored-entanglement times of
    L-B6a/L-B6b; the placement variant has no tower and no spreading.
 5. Yukawa gravity V = -G m1 m2 (1 + alpha e^{-r/lambda}) / r: fractional
    deviations at r = 52 um and 3 mm for lambda = 30 um, alpha = 1.
 6. Allowed window: R < 30 um (L-C1a) hence L < 47 um if qD couples to
    gravity universally; the free-motion variant is excluded at every R;
    inside the window no tabletop signature remains.
 7. Spin statistics are those of M1: S = 2 sqrt 2, marginals 1/2 (see R01).

SI units throughout; energies are returned in joules unless the name says eV.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import frames, qm

# ------------------------------------------------------------- constants
HBAR = 1.054571817e-34   # J s
C = 2.99792458e8         # m / s
M_E = 9.1093837e-31      # kg
E_CHARGE = 1.602176634e-19  # C  (1 eV = this many joules)
G = 6.67430e-11          # m^3 kg^-1 s^-2
K_B = 1.380649e-23       # J / K

EV = E_CHARGE
KT_ROOM = 0.025 * EV     # ~ 290 K

# Ledger numbers used (registers/constraints.md)
L_C1A_LAMBDA_MAX = 38.6e-6   # m, |alpha| = 1 Yukawa, 95 % CL (Lee et al. 2020)
L_C1A_R_STAR_MAX = 30e-6     # m, largest extra dimension (Lee et al. 2020)
L_C1B_LAMBDA_EXCLUDED = 48e-6  # m, ISL holds down to this lambda (Tan et al. 2020)
L_B6A_STORED_S = (2.0, 18.0)   # s, F_min > 0.5 window (Haeffner et al. 2005)
L_B6A_DECAY_S = 34.0           # s
L_B6B_STORED_S = 1e3           # s, logical Bell states (Zhang et al. 2026)
L_A6D_S = 2.731                # Steinacker et al. 2025, quantum-dot electron spins
L_A6E_S = 2.70                 # Dehollain et al. 2016, donor electron + nuclear spin


# --------------------------------------------- claim 1: physical coordinate
def physical_coordinate(L: float, z) -> np.ndarray:
    """Z = L z."""
    return L * np.asarray(z, dtype=float)


def to_frame_physical(k: int, l: int, a: float, x, y, Z, L: float):
    """Frame map k -> l written in the physical qD coordinate Z = L z."""
    x2, y2, z2 = frames.to_frame(k, l, a, x, y, np.asarray(Z, dtype=float) / L)
    return x2, y2, L * z2


def lab_tracks_physical(a: float, y, L: float):
    """Lab tracks of A (Z = -L) and B (Z = +L), each at rest at x = 0 in its
    own frame, mapped to the lab with the physical coordinate."""
    y = np.asarray(y, dtype=float)
    xA, _, _ = to_frame_physical(frames.A, frames.LAB, a, 0 * y, y, -L * np.ones_like(y), L)
    xB, _, _ = to_frame_physical(frames.B, frames.LAB, a, 0 * y, y, +L * np.ones_like(y), L)
    return xA, xB


def max_L_in_circle(R: float) -> float:
    """Largest L for which Z = -L and Z = +L fit on a circle of radius R
    (they are then antipodal): L = pi R / 2."""
    return float(np.pi * R / 2)


# -------------------------------------------- claims 2-3: excitation towers
def kk_massless_energy(n, R) -> np.ndarray:
    """E_n = n hbar c / R  (Kaluza-Klein tower of a massless field)."""
    return np.asarray(n, dtype=float) * HBAR * C / np.asarray(R, dtype=float)


def ring_energy(n, R, m: float = M_E) -> np.ndarray:
    """E_n = n^2 hbar^2 / (2 m R^2)  (particle on a ring, non-relativistic)."""
    n = np.asarray(n, dtype=float)
    R = np.asarray(R, dtype=float)
    return n * n * HBAR * HBAR / (2 * m * R * R)


def kk_massive_excitation(n, R, m: float = M_E) -> np.ndarray:
    """Exact excitation of a massive KK mode above the ground state:
    sqrt((m c^2)^2 + (n hbar c / R)^2) - m c^2, written in the cancellation-free
    form (n hbar c / R)^2 / (sqrt(...) + m c^2)."""
    p_c = kk_massless_energy(n, R)
    mc2 = m * C * C
    return p_c * p_c / (np.sqrt(mc2 * mc2 + p_c * p_c) + mc2)


def crossover_radius(m: float = M_E) -> float:
    """R at which the massless-KK and ring first excitations coincide:
    hbar c / R = hbar^2 / (2 m R^2)  ->  R = hbar / (2 m c)."""
    return float(HBAR / (2 * m * C))


def radius_for_massless(E: float) -> float:
    """R at which the first massless-KK excitation equals E."""
    return float(HBAR * C / E)


def radius_for_ring(E: float, m: float = M_E) -> float:
    """R at which the first ring excitation equals E."""
    return float(np.sqrt(HBAR * HBAR / (2 * m * E)))


def tower_table(R=None) -> np.ndarray:
    """Columns: R [m], E_1 massless-KK [eV], E_1 ring [eV], E_1 massive-KK [eV]."""
    if R is None:
        R = np.logspace(-12, -4, 9)
    R = np.asarray(R, dtype=float)
    return np.column_stack([R, kk_massless_energy(1, R) / EV, ring_energy(1, R) / EV,
                            kk_massive_excitation(1, R) / EV])


def thresholds() -> dict:
    """Radii at which E_1 crosses 1 eV, kT (0.025 eV) and 1 keV, both towers."""
    out = {}
    for name, f in (("massless", radius_for_massless), ("ring", radius_for_ring)):
        out[f"{name}_R_at_1eV"] = f(1.0 * EV)
        out[f"{name}_R_at_kT"] = f(KT_ROOM)
        out[f"{name}_R_at_1keV"] = f(1e3 * EV)
    return out


# ------------------------------------------- claim 4: spreading in qD
def spreading_time(R, delta=None, m: float = M_E) -> np.ndarray:
    """Time for a packet of initial width delta (default: the whole radius R,
    the most generous choice) to spread across a circle of radius R:
    tau ~ m R delta / hbar, from v_spread ~ hbar / (m delta)."""
    R = np.asarray(R, dtype=float)
    delta = R if delta is None else np.asarray(delta, dtype=float)
    return m * R * delta / HBAR


def placement_persistence_ratio(R) -> np.ndarray:
    """Shortest ledger stored-entanglement time (L-B6a lower edge, 2 s)
    divided by the spreading time: >> 1 means free motion in qD cannot keep
    the pair placed at Z = +-L for as long as entanglement is observed to last."""
    return L_B6A_STORED_S[0] / spreading_time(R)


# ------------------------------------------------ claim 5: Yukawa gravity
def yukawa_potential(r, m1: float, m2: float, alpha: float, lam: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return -G * m1 * m2 * (1 + alpha * np.exp(-r / lam)) / r


def newton_potential(r, m1: float, m2: float) -> np.ndarray:
    return -G * m1 * m2 / np.asarray(r, dtype=float)


def yukawa_force(r, m1: float, m2: float, alpha: float, lam: float) -> np.ndarray:
    """Attractive force magnitude -dV/dr."""
    r = np.asarray(r, dtype=float)
    return G * m1 * m2 / r**2 * (1 + alpha * np.exp(-r / lam) * (1 + r / lam))


def fractional_potential_deviation(r, alpha: float, lam: float) -> np.ndarray:
    """(V_Yukawa - V_Newton) / V_Newton = alpha e^{-r/lambda} (closed form; the
    subtraction form underflows for r >> lambda)."""
    r = np.asarray(r, dtype=float)
    return alpha * np.exp(-r / lam)


def fractional_force_deviation(r, alpha: float, lam: float) -> np.ndarray:
    """(F_Yukawa - F_Newton) / F_Newton = alpha e^{-r/lambda} (1 + r/lambda)."""
    r = np.asarray(r, dtype=float)
    return alpha * np.exp(-r / lam) * (1 + r / lam)


def deviations_by_subtraction(r, m1: float, m2: float, alpha: float, lam: float):
    """The same two deviations computed by subtracting the potentials/forces
    (consistency check of the closed forms where they do not underflow)."""
    r = np.asarray(r, dtype=float)
    vN = newton_potential(r, m1, m2)
    fN = G * m1 * m2 / r**2
    return ((yukawa_potential(r, m1, m2, alpha, lam) - vN) / vN,
            (yukawa_force(r, m1, m2, alpha, lam) - fN) / fN)


def yukawa_sensitivity(lam: float = 30e-6, alpha: float = 1.0, r=(52e-6, 1e-3, 3e-3)) -> np.ndarray:
    """Columns: r [m], potential deviation, force deviation."""
    r = np.asarray(r, dtype=float)
    return np.column_stack([r, fractional_potential_deviation(r, alpha, lam),
                            fractional_force_deviation(r, alpha, lam)])


# --------------------------------------------------- claim 6: the window
def allowed_window() -> dict:
    """Bounds on R and L for each sub-variant of M3."""
    th = thresholds()
    R_grav = L_C1A_R_STAR_MAX
    out = {
        # universal gravitational coupling: ledger bound on the dimension
        "R_max_gravity": R_grav,
        "L_max_gravity": max_L_in_circle(R_grav),
        "lambda_max_alpha1": L_C1A_LAMBDA_MAX,
        # free motion: tower must sit above 1 keV to escape tested energies
        "R_max_free_massless_1keV": th["massless_R_at_1keV"],
        "R_max_free_ring_1keV": th["ring_R_at_1keV"],
        # ... and even then the placement decays in the spreading time
        "tau_spread_at_R_gravity": float(spreading_time(R_grav)),
        "tau_spread_at_R_free": float(spreading_time(th["massless_R_at_1keV"])),
        "persistence_ratio_at_R_gravity": float(placement_persistence_ratio(R_grav)),
        # tabletop signature left inside the window (placement variant)
        "force_deviation_1mm_at_R_gravity": float(fractional_force_deviation(1e-3, 1.0, R_grav)),
        "ring_E1_eV_at_R_gravity": float(ring_energy(1, R_grav) / EV),
    }
    out["free_motion_excluded"] = out["persistence_ratio_at_R_gravity"] > 1 and \
        placement_persistence_ratio(th["massless_R_at_1keV"]) > 1
    return out


# ------------------------------------------------- claim 7: spin (as M1)
def singlet_statistics() -> dict:
    a, ap, b, bp = qm.chsh_optimal_settings()
    psi = qm.singlet()
    S = qm.chsh(psi, a, ap, b, bp)
    margs = [qm.marginal_probability(psi, s, +1, w) for w in ("A", "B") for s in (a, ap, b, bp)]
    return {"S": float(abs(S)), "marginals": np.array(margs),
            "ledger_S_within_bound": bool(max(L_A6D_S, L_A6E_S) <= qm.TSIRELSON)}


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    print("claim 1: lab tracks at y=1, a=2 for L = 1e-6, 1, 1e3 m:")
    for L in (1e-6, 1.0, 1e3):
        print("  L =", L, lab_tracks_physical(2.0, np.array([1.0]), L))
    print("  L_max on circle R = 30 um:", max_L_in_circle(30e-6))
    print("claims 2-3: R [m], E1 massless-KK [eV], E1 ring [eV], E1 massive-KK [eV]")
    for row in tower_table():
        print("  %.0e  %.3e  %.3e  %.3e" % tuple(row))
    print("  thresholds:", {k: "%.3e" % v for k, v in thresholds().items()})
    print("  crossover radius:", crossover_radius())
    print("claim 4: spreading time at R = 30 um:", spreading_time(30e-6), "s;",
          "at 1.97e-10 m:", spreading_time(1.97e-10), "s")
    print("claim 5: r, dV/V, dF/F for lambda = 30 um, alpha = 1")
    print(yukawa_sensitivity())
    print("claim 6:", allowed_window())
    print("claim 7:", singlet_statistics())
