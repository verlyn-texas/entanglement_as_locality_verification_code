"""R07 — realistic (uncertainty-limited) position tracking of the two electrons.

Mechanism M5: each electron's centre-of-mass motion along the lab axis x is a
free particle under *continuous weak position measurement* of strength k;
the spin singlet is a separate factor (spin (x) position) that the position
measurement never touches; the particle frames 1, 2 of ``transformation.md``
are attached to the lab's *conditional-mean* tracks <x>_A(y), <x>_B(y).

Convention (Jacobs & Steck, Contemp. Phys. 47, 279 (2006), with hbar
restored).  The stochastic master equation for a measurement of x with
strength k (units m^-2 s^-1: inverse variance resolved per unit time) is

    d rho = -(i/hbar)[H, rho] dt - k [x,[x, rho]] dt
            + sqrt(2k) (x rho + rho x - 2<x> rho) dW,

with record  dy = <x> dt + dW / sqrt(8k),  E[dW^2] = dt.  For H = p^2/2m and a
Gaussian state the conditional moments obey (V_x = Var x, V_p = Var p,
C = Cov(x,p) = <xp+px>/2 - <x><p>):

    d<x> = (<p>/m) dt + sqrt(8k) V_x dW,      d<p> = sqrt(8k) C dW,
    dV_x/dt = 2C/m - 8k V_x^2,   dV_p/dt = 2 hbar^2 k - 8k C^2,
    dC/dt   = V_p/m - 8k V_x C.

Integrated by Euler–Maruyama with a fixed seed; the two electrons are
independent runs with <p>(0) = -p0 (A) and +p0 (B), so the ideal lab tracks
are x = -/+ y/a with 1/a = p0/m.

Claims (numbered as in rows/R07_realistic_tracking.md):
 1. Steady state of the covariance equations (closed form):
    C = hbar/2, V_x = sqrt(hbar/(8 m k)), V_p = hbar sqrt(2 hbar m k), and the
    uncertainty product D = V_x V_p - C^2 = hbar^2/4 (minimum uncertainty).
    The code reproduces it from a mixed initial state.
 2. dD/dt = -8k V_x (D - hbar^2/4): D relaxes monotonically to hbar^2/4 from
    above and never crosses it -> Heisenberg is respected at every step; a
    pure initial state keeps D = hbar^2/4 along the whole stochastic run.
 3. Momentum diffusion: ensemble <p^2> - p0^2 - V_p(0) = 2 hbar^2 k t exactly
    (d(V_p + E[dp^2])/dt = 2 hbar^2 k step by step), heating hbar^2 k/m.
 4. Precision of the separation speed 2/a fitted to the tracked separation:
    Var(slope) = (2 hbar/m)(1.2/T + gamma/2) + (52/35) hbar^2 k T/m^2 with
    gamma = sqrt(8 k hbar/m); a fit to the raw record (a smoother) drops the
    1.2/T jitter term and adds 1.5/(k T^3) of white record noise.
 5. In A's frame (path_in_frame with p_i = deviation of the track from -/+ y/a)
    the tracked B has zero-mean separation with
    Var = 2[(hbar/m) t + (8k V_x C/m) t^2 + (2 hbar^2 k/m^2) t^3/3].
 6. Spin: CHSH S = 2 sqrt 2 before and after tracking; the position
    dissipator leaves the reduced spin state exactly unchanged; a
    spin-dependent measurement operator (x sigma_z, a field gradient) does not.
 7. Realistic numbers (v0 = 1e4 m/s): table of k, sqrt(V_x), gamma, heating,
    and the relative error of 2/a.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import frames, qm

M_E = 9.1093837e-31        # kg
HBAR = 1.054571817e-34     # J s
EV = 1.602176634e-19       # J
V0 = 1.0e4                 # m/s, each electron's lab speed (non-relativistic)
P0 = M_E * V0              # kg m/s
A_PARAM = 1.0 / V0         # a = 1/(half the separation speed): lab sees -/+ y/a

KS_TABLE = (1e19, 1e21, 1e23, 1e25, 1e27)


# ------------------------------------------------------------ closed forms
def steady_state(k: float, m: float = M_E, hbar: float = HBAR) -> dict:
    """Fixed point of the covariance equations (claim 1)."""
    C = hbar / 2
    Vx = np.sqrt(hbar / (8 * m * k))
    Vp = hbar * np.sqrt(2 * hbar * m * k)
    return {"Vx": Vx, "Vp": Vp, "C": C, "D": Vx * Vp - C * C,
            "gamma": relaxation_rate(k, m, hbar)}


def relaxation_rate(k: float, m: float = M_E, hbar: float = HBAR) -> float:
    """gamma = 8 k V_x^ss = sqrt(8 k hbar / m): rate at which D -> hbar^2/4."""
    return float(np.sqrt(8 * k * hbar / m))


def heating_rate(k: float, m: float = M_E, hbar: float = HBAR) -> float:
    """d<E>/dt = hbar^2 k / m (J/s) — backaction heating (claim 3)."""
    return hbar * hbar * k / m


def uncertainty_product(Vx, Vp, C):
    return Vx * Vp - C * C


def covariance_rhs(Vx, Vp, C, k, m=M_E, hbar=HBAR):
    return (2 * C / m - 8 * k * Vx * Vx,
            2 * hbar * hbar * k - 8 * k * C * C,
            Vp / m - 8 * k * Vx * C)


def slope_variance_closed_form(k: float, T: float, m: float = M_E, hbar: float = HBAR,
                               record: bool = False) -> float:
    """Variance of the least-squares slope of the tracked separation
    s(t) = <x>_B - <x>_A over [0, T], covariances at steady state (claim 4).
    Terms: jitter of <x> (diffusion hbar/m per electron), correlated
    <x>–<p> kicks (8 k V_x C/m, T-independent), momentum diffusion
    (2 hbar^2 k per electron); ``record=True``: fit to the raw record."""
    ss = steady_state(k, m, hbar)
    jitter = 1.2 * (hbar / m) / T
    cross = 8 * k * ss["Vx"] * ss["C"] / m
    momentum = (13 / 35) * (2 * hbar * hbar * k / (m * m)) * T
    if not record:
        return float(2 * (jitter + cross + momentum))
    # The record noise is the innovation that drives <x>, so a line through
    # the raw record is a *smoother*: its correlation with the <x> jitter
    # (-2.4 hbar/(mT) in total) cancels the jitter term exactly, and the white
    # record noise (spectral density 1/(8k)) adds 12/(8 k T^3).
    return float(2 * (cross + momentum) + 1.5 / (k * T ** 3))


def separation_variance_closed_form(k: float, t, m: float = M_E, hbar: float = HBAR):
    """Var[s(t) - 2t/a] of the tracked separation about its ideal value
    (claim 5), covariances at steady state."""
    ss = steady_state(k, m, hbar)
    t = np.asarray(t, dtype=float)
    return 2 * ((hbar / m) * t + (8 * k * ss["Vx"] * ss["C"] / m) * t ** 2
                + (2 * hbar * hbar * k / (m * m)) * t ** 3 / 3)


# ------------------------------------------------------------- integrators
def integrate_covariances(Vx0: float, Vp0: float, C0: float, k: float, T: float,
                          n_steps: int, m: float = M_E, hbar: float = HBAR) -> dict:
    """RK4 integration of the (deterministic) covariance equations.  These
    carry no noise, so a fourth-order step is used to keep the exact
    invariants (D = hbar^2/4 on a pure state) to round-off; the stochastic
    means are stepped by Euler–Maruyama in ``simulate_pair``."""
    dt = T / n_steps
    Vx = np.empty(n_steps + 1); Vp = np.empty(n_steps + 1); C = np.empty(n_steps + 1)
    Vx[0], Vp[0], C[0] = Vx0, Vp0, C0
    f = lambda y: np.array(covariance_rhs(y[0], y[1], y[2], k, m, hbar))
    for i in range(n_steps):
        y = np.array([Vx[i], Vp[i], C[i]])
        k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        Vx[i + 1], Vp[i + 1], C[i + 1] = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    t = dt * np.arange(n_steps + 1)
    return {"t": t, "Vx": Vx, "Vp": Vp, "C": C, "D": uncertainty_product(Vx, Vp, C)}


def relaxation_run(k: float = 1e19, widen: float = 4.0, n_per_gamma: int = 100,
                   n_gammas: float = 30.0) -> dict:
    """Start from a *mixed* Gaussian (V_x widened by ``widen``, V_p at its
    steady value, C = 0, so D = widen * hbar^2/2 > hbar^2/4) and relax."""
    ss = steady_state(k)
    T = n_gammas / ss["gamma"]
    out = integrate_covariances(widen * ss["Vx"], ss["Vp"], 0.0, k, T, int(n_per_gamma * n_gammas))
    out["steady"] = ss
    out["D0"] = widen * ss["Vx"] * ss["Vp"]
    return out


def simulate_pair(k: float, T: float, n_steps: int = 2000, n_traj: int = 1, seed: int = 7,
                  start: str = "steady", sigma0: float | None = None,
                  m: float = M_E, hbar: float = HBAR, p0: float = P0) -> dict:
    """Euler–Maruyama run of both electrons, ``n_traj`` independent
    realisations each.  Returns the conditional means xA, xB (shape
    (n_traj, n_steps+1)), the momentum deviations dpA = <p>_A + p0,
    dpB = <p>_B - p0, the records rA, rB (dy/dt samples), and the shared
    deterministic covariances.  ``start='steady'`` puts the covariances at the
    fixed point; ``start='pure'`` uses a minimum-uncertainty packet of width
    sigma0 (V_x = sigma0^2, V_p = hbar^2/(4 sigma0^2), C = 0)."""
    dt = T / n_steps
    if start == "steady":
        ss = steady_state(k, m, hbar)
        Vx0, Vp0, C0 = ss["Vx"], ss["Vp"], ss["C"]
    elif start == "pure":
        Vx0, Vp0, C0 = sigma0 ** 2, hbar ** 2 / (4 * sigma0 ** 2), 0.0
    else:
        raise ValueError(start)
    cov = integrate_covariances(Vx0, Vp0, C0, k, T, n_steps, m, hbar)
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((2, n_traj, n_steps)) * np.sqrt(dt)
    x = np.zeros((2, n_traj, n_steps + 1))
    dp = np.zeros((2, n_traj, n_steps + 1))
    rec = np.zeros((2, n_traj, n_steps))
    pmean0 = np.array([-p0, p0])[:, None]
    s8k = np.sqrt(8 * k)
    for i in range(n_steps):
        p_mean = pmean0 + dp[:, :, i]
        x[:, :, i + 1] = x[:, :, i] + (p_mean / m) * dt + s8k * cov["Vx"][i] * dW[:, :, i]
        dp[:, :, i + 1] = dp[:, :, i] + s8k * cov["C"][i] * dW[:, :, i]
        rec[:, :, i] = x[:, :, i] + dW[:, :, i] / (s8k * dt)
    return {"t": cov["t"], "dt": dt, "k": k, "xA": x[0], "xB": x[1], "dpA": dp[0], "dpB": dp[1],
            "rA": rec[0], "rB": rec[1], "Vx": cov["Vx"], "Vp": cov["Vp"], "C": cov["C"],
            "D": cov["D"], "p0": p0, "m": m}


# --------------------------------------------------------------- analyses
def ls_slope(t, s):
    """Least-squares slope of s (…, n) against t (n), vectorised."""
    tc = t - t.mean()
    sc = s - s.mean(axis=-1, keepdims=True)
    return (sc * tc).sum(axis=-1) / (tc * tc).sum()


def fitted_separation_speed(run: dict, use_record: bool = False) -> np.ndarray:
    """Slope of the tracked separation x_B - x_A per trajectory (claim 4)."""
    if use_record:
        t = run["t"][:-1]
        s = run["rB"] - run["rA"]
    else:
        t = run["t"]
        s = run["xB"] - run["xA"]
    return ls_slope(t, s)


def momentum_diffusion(run: dict) -> dict:
    """Ensemble <p^2> - p0^2 - V_p(0) versus 2 hbar^2 k t (claim 3)."""
    t = run["t"]
    Edp2 = 0.5 * ((run["dpA"] ** 2).mean(axis=0) + (run["dpB"] ** 2).mean(axis=0))
    ensemble = run["Vp"] - run["Vp"][0] + Edp2
    # deterministic bookkeeping: dV_p/dt + 8k C^2 = 2 hbar^2 k at every step
    rhs_Vp = 2 * HBAR ** 2 * run["k"] - 8 * run["k"] * run["C"] ** 2
    identity = rhs_Vp + 8 * run["k"] * run["C"] ** 2
    return {"t": t, "ensemble": ensemble, "theory": 2 * HBAR ** 2 * run["k"] * t,
            "identity": identity, "expected_identity": 2 * HBAR ** 2 * run["k"]}


def track_deviation(run: dict, a: float = A_PARAM, traj: int = 0):
    """The paths p_A, p_B of transformation.md §4: deviation of the tracked
    lab path from the ideal -/+ y/a, as callables in each particle's own
    frame (frame 1 for A, frame 2 for B)."""
    t, xA, xB = run["t"], run["xA"][traj], run["xB"][traj]
    pA = lambda y: np.interp(y, t, xA) + np.asarray(y) / a
    pB = lambda y: np.interp(y, t, xB) - np.asarray(y) / a
    return pA, pB


def separation_in_frames(run: dict, a: float = A_PARAM, traj: int = 0) -> dict:
    """Tracked separation x_B - x_A seen in the lab and in A's frame, via
    frames.path_in_frame (claim 5, single trajectory)."""
    t = run["t"]
    pA, pB = track_deviation(run, a, traj)
    out = {}
    for k_frame, name in ((frames.LAB, "lab"), (frames.A, "A"), (frames.B, "B")):
        xa = frames.path_in_frame(frames.A, k_frame, a, pA, t)
        xb = frames.path_in_frame(frames.B, k_frame, a, pB, t)
        out[name] = xb - xa
    return out


def separation_in_A_frame_ensemble(run: dict, a: float = A_PARAM) -> np.ndarray:
    """x_B - x_A in A's frame for every trajectory: the tracked separation
    minus 2y/a (what path_in_frame gives, see separation_in_frames)."""
    return run["xB"] - run["xA"] - 2 * run["t"] / a


# -------------------------------------------------------------------- spin
def spin_chsh_before_after(k: float = 1e19, T: float = 1e-7) -> dict:
    """The state is rho_spin (x) Gaussian(position).  The SME acts on the
    position factor only, so the spin factor carried through a tracking run
    is the same object (claim 6)."""
    a, ap, b, bp = qm.chsh_optimal_settings()
    rho_spin = qm.dm(qm.singlet())
    S_before = qm.chsh(rho_spin, a, ap, b, bp)
    run = simulate_pair(k, T, n_steps=200, n_traj=1)
    rho_spin_after = rho_spin  # position SME: no operator acts on the spin factor
    S_after = qm.chsh(rho_spin_after, a, ap, b, bp)
    return {"S_before": float(S_before), "S_after": float(S_after),
            "concurrence": qm.concurrence(rho_spin_after), "final_D": float(run["D"][-1])}


def _dissipator(L, rho):
    LdL = L.conj().T @ L
    return L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)


def reduced_spin_change(n_pos: int = 4, coupling: str = "position", seed: int = 3) -> dict:
    """Toy check with a discretised position (n_pos levels): apply the
    measurement dissipator k[x,[x,.]] = -2k D[x] to rho_spin (x) rho_pos and
    trace out position.  ``coupling='position'``: L = 1 (x) x -> the reduced
    spin state does not change.  ``coupling='gradient'``: L = sigma_z^A (x) x
    (sensor reads a spin-dependent position, i.e. a field gradient) -> it does."""
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n_pos, n_pos)) + 1j * rng.standard_normal((n_pos, n_pos))
    rho_pos = G @ G.conj().T
    rho_pos /= np.trace(rho_pos)
    x_op = np.diag(np.linspace(-1.5, 1.5, n_pos)).astype(complex)
    rho_spin = qm.dm(qm.singlet())
    rho = np.kron(rho_spin, rho_pos)
    if coupling == "position":
        L = np.kron(np.eye(4), x_op)
    elif coupling == "gradient":
        L = np.kron(qm.kron(qm.SZ, qm.I2), x_op)
    else:
        raise ValueError(coupling)
    drho = 2.0 * _dissipator(L, rho)                # -k[x,[x,rho]] = 2k D[x]rho, k = 1
    d_spin = np.trace(drho.reshape(4, n_pos, 4, n_pos), axis1=1, axis2=3)
    a, ap, b, bp = qm.chsh_optimal_settings()
    eps = 0.05                                      # one small Lindblad step
    rho_spin_new = rho_spin + eps * d_spin
    return {"norm_change": float(np.linalg.norm(d_spin)),
            "S_after_step": float(qm.chsh(rho_spin_new, a, ap, b, bp))}


def spin_dephasing_rate_ratio(n_pos: int = 81, width: float = 1.0) -> float:
    """Round-4 item G1.  With the Eq. (12) convention (measurement term
    -Gamma_m [A, [A, rho]]) and a sensor reading the spin-dependent position
    A = sigma_z (x) x, the reduced-spin coherence of a packet with second
    moment <x^2> decays initially at 4 Gamma_m <x^2> (the two spin branches
    see +-x, a separation 2x).  Returns -(d rho_01/dt) / (Gamma_m <x^2> rho_01)
    on a discretised Gaussian packet: 4."""
    xs = np.linspace(-6 * width, 6 * width, n_pos)
    psi = np.exp(-xs**2 / (4 * width**2)).astype(complex)
    psi /= np.linalg.norm(psi)
    rho_pos = np.outer(psi, psi.conj())
    x2 = float(np.real(np.sum(xs**2 * np.abs(psi) ** 2)))
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    rho_spin = np.outer(plus, plus.conj())
    rho = np.kron(rho_spin, rho_pos)
    A = np.kron(np.diag([1.0, -1.0]).astype(complex), np.diag(xs).astype(complex))
    drho = -(A @ (A @ rho - rho @ A) - (A @ rho - rho @ A) @ A)      # -[A,[A,rho]], Gamma_m = 1
    d_spin = np.trace(drho.reshape(2, n_pos, 2, n_pos), axis1=1, axis2=3)
    return float(-np.real(d_spin[0, 1]) / (x2 * np.real(rho_spin[0, 1])))


# ------------------------------------------------------------------- table
def realistic_table(ks=KS_TABLE, T: float = 1e-7, m: float = M_E, v0: float = V0) -> list[dict]:
    """(k, sqrt V_x^ss, gamma, heating, relative error of the fitted 2/a)."""
    rows = []
    for k in ks:
        ss = steady_state(k, m)
        rows.append({"k": k, "sqrtVx_m": float(np.sqrt(ss["Vx"])), "gamma": ss["gamma"],
                     "heating_eV_per_s": heating_rate(k, m) / EV,
                     "T": T, "rel_err_2_over_a": float(np.sqrt(slope_variance_closed_form(k, T, m)) / (2 * v0)),
                     "t_double_KE_s": 0.5 * m * v0 ** 2 / heating_rate(k, m)})
    return rows


def optimal_run_time(k: float, m: float = M_E, hbar: float = HBAR, v0: float = V0) -> dict:
    """T* minimising the closed-form slope variance (jitter ~ 1/T against
    momentum diffusion ~ T): T* = sqrt(21 m / (13 hbar k)); returns T* and
    the relative error of 2/a there."""
    T = float(np.sqrt(21 * m / (13 * hbar * k)))
    return {"T_opt": T, "rel_err_min": float(np.sqrt(slope_variance_closed_form(k, T, m, hbar)) / (2 * v0))}


def precision_grid(ks=(1e19, 1e21, 1e23), Ts=(1e-8, 1e-7, 1e-6), m: float = M_E, v0: float = V0):
    """Relative rms error of the fitted 2/a (closed form) on a (k, T) grid."""
    return np.array([[np.sqrt(slope_variance_closed_form(k, T, m)) / (2 * v0) for T in Ts] for k in ks])


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    ss = steady_state(1e19)
    print("steady state k=1e19:", {kk: f"{v:.4e}" for kk, v in ss.items()}, "hbar^2/4 =", HBAR ** 2 / 4)
    rel = relaxation_run(1e19)
    print("relaxation: D0/(hbar^2/4) =", rel["D0"] / (HBAR ** 2 / 4), " final D/(hbar^2/4) =",
          rel["D"][-1] / (HBAR ** 2 / 4), " min D/(hbar^2/4) =", rel["D"].min() / (HBAR ** 2 / 4))
    run = simulate_pair(1e19, 1e-7, n_steps=2000, n_traj=2000)
    md = momentum_diffusion(run)
    print("momentum diffusion: ensemble/theory at T =", md["ensemble"][-1] / md["theory"][-1])
    sl = fitted_separation_speed(run)
    print("fitted 2/a: mean", sl.mean(), " rms err", sl.std(), " closed form",
          np.sqrt(slope_variance_closed_form(1e19, 1e-7)), " relative", sl.std() / (2 * V0))
    slr = fitted_separation_speed(run, use_record=True)
    print("record fit rms", slr.std(), " closed form", np.sqrt(slope_variance_closed_form(1e19, 1e-7, record=True)))
    sepA = separation_in_A_frame_ensemble(run)
    print("A-frame separation at T: mean", sepA[:, -1].mean(), " rms", sepA[:, -1].std(),
          " closed form", np.sqrt(separation_variance_closed_form(1e19, 1e-7)))
    print(separation_in_frames(run)["lab"][-1], separation_in_frames(run)["A"][-1])
    print(spin_chsh_before_after())
    print(reduced_spin_change(coupling="position"), reduced_spin_change(coupling="gradient"))
    for row in realistic_table():
        print({kk: (f"{v:.3g}" if isinstance(v, float) else v) for kk, v in row.items()})
    print(precision_grid())
    print("optimal run time k=1e19:", optimal_run_time(1e19))
