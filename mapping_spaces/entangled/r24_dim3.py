"""R24 — the charts in 3+1 dimensions (round-3 revision, handoff3 A2/A3/B3).

Round 3 established (referee B's check script, 15/15, ported here) that the
"open structural obstacle" v3 recorded for non-collinear boosts does not
exist, and that the paper's Appendix on the 3+1 charts is a computation, not
a research programme:

* Galilean base form: with VECTOR-valued potentials pinned by P1 verbatim,
  composition, inverses, partner-at-rest (fraction 1-w of the relative
  velocity) and lab pinning all hold in 3-D — Galilean boosts commute and do
  not rotate spin.
* Covariant variant: with GROUP-valued potentials Lambda_k(z) in SO+(1,3),
  pinned by the same rule, and frame maps defined as right quotients
  T_kl(z) = Lambda_l(z) Lambda_k(z)^-1, composition T_lm T_kl = T_km and
  inverses hold identically by group multiplication.  Partners sit at x'=0
  with frame time = proper time; a w<1 partner moves at (1-w) x the
  relativistic relative velocity; the collinear limit recovers the scalar
  rapidities of Sec. 3.7.1.  Maps between unrelated particles' frames carry
  Wigner rotations (up to ~24 deg here) — a fixed local unitary per particle
  on sharp tracks, changing no cut negativity and no statistic.
* Light-cone rule (Eq. 10): with the Euclidean norm the kappa-clocks, the
  cone sign and the ordering are invariant under non-collinear boosts, and
  the disagreement fraction (R-1)/2R holds for perpendicular velocities.
* Photons: kappa = sqrt(1-v^2)(t-t0) vanishes identically on null
  worldlines (every photon pair is on the tie set), and an affine-parameter
  clock is not boost-invariant — the order of two back-to-back photons flips
  under boosts (Doppler rescaling (1+b)/(1-b)), while a massive pair at the
  same lab times never flips.
* P2 normalization: w in [0,1] is a one-line theorem — each singleton cut
  {i}|rest is a separating cut with negativity <= (d_i - 1)/2, so the
  min-cut negativity never exceeds (d_min - 1)/2.

Every claim of paper 1's Appendix C ("The charts in 3+1 dimensions") and of
the rewritten Sec. 3.7.4 numerics is backed by tests/test_entangled_r24.py.
"""
from __future__ import annotations

import numpy as np

ETA = np.diag([1.0, -1.0, -1.0, -1.0])


# ---------------------------------------------------------------- Galilean 3+1
def galilean_potentials(velocities, heights, weights):
    """Pinned vector potentials of P1 in 3-D.

    velocities: (N, 3) lab velocities; heights: (N,) label heights (nonzero);
    weights: (N, N) symmetric locality weights.  Returns a list of N+1 dicts
    {height: 3-vector} — one per particle frame, the lab frame (index N) last.
    Heights pinned: 0.0 (the lab) and every particle height.
    """
    velocities = np.asarray(velocities, float)
    n = len(velocities)
    pots = []
    for i in range(n):
        vals = {0.0: -velocities[i]}
        for j in range(n):
            if j == i:
                vals[heights[j]] = np.zeros(3)
            else:
                vals[heights[j]] = (1 - weights[i][j]) * (velocities[j] - velocities[i])
        pots.append(vals)
    lab = {0.0: np.zeros(3)}
    for j in range(n):
        lab[heights[j]] = velocities[j]
    pots.append(lab)
    return pots


def galilean_map(pots, k, l, height, x, t):
    """Frame k -> frame l at a pinned height (Eq. 2, componentwise)."""
    return np.asarray(x, float) + (pots[l][height] - pots[k][height]) * t


# ------------------------------------------------------------- covariant 3+1
def boost(v):
    """Pure boost B(v) in SO+(1,3): rest-frame coords -> coords moving at v (c=1)."""
    v = np.asarray(v, float)
    b2 = v @ v
    L = np.eye(4)
    if b2 < 1e-30:
        return L
    g = 1.0 / np.sqrt(1.0 - b2)
    L[0, 0] = g
    L[0, 1:] = g * v
    L[1:, 0] = g * v
    L[1:, 1:] = np.eye(3) + (g - 1.0) * np.outer(v, v) / b2
    return L


def lorentz_inv(L):
    """Inverse of a Lorentz matrix via the metric."""
    return ETA @ L.T @ ETA


def rel_velocity(vk, vj):
    """Velocity 3-vector of a particle moving at vj, seen from the rest frame of vk."""
    U = boost(vj)[:, 0]  # 4-velocity of j in the lab
    Up = lorentz_inv(boost(vk)) @ U
    return Up[1:] / Up[0]


def lambda_potential(k, height_index, velocities, weights):
    """Group-valued potential Lambda_k(z): lab -> frame k at height of particle
    `height_index` ('lab' for the lab height).  k = 'lab' or a particle index."""
    if k == "lab":
        return np.eye(4)
    if height_index in ("lab", k):
        return lorentz_inv(boost(velocities[k]))
    u = (1 - weights[k][height_index]) * rel_velocity(velocities[k], velocities[height_index])
    return boost(u) @ lorentz_inv(boost(velocities[height_index]))


def frame_map(k, l, height_index, velocities, weights):
    """T_kl(z) = Lambda_l(z) Lambda_k(z)^-1 — the right quotient."""
    return lambda_potential(l, height_index, velocities, weights) @ lorentz_inv(
        lambda_potential(k, height_index, velocities, weights)
    )


def rotation_part(L):
    """Rotation block of the polar-like split L = B(u) R (u = velocity of the image)."""
    u = L[1:, 0] / L[0, 0]
    R = lorentz_inv(boost(u)) @ L
    return R[1:, 1:]


def rotation_angle_deg(L):
    R = rotation_part(L)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


# ---------------------------------------------------- light-cone rule in 3+1
def crossing_time(x0, t0, v, xT, tT):
    """Unique t with |x0 + v (t - t0) - xT| = t - tT and t >= tT (v < 1)."""
    x0, v, xT = (np.asarray(a, float) for a in (x0, v, xT))
    d = x0 - xT
    a0 = t0 - tT
    A = v @ v - 1.0
    B = 2.0 * (d @ v - a0)
    C = d @ d - a0**2
    roots = np.roots([A, B, C])
    roots = np.real(roots[np.isreal(roots)])
    s = [r for r in roots if r + a0 >= -1e-12]
    s = max(s) if len(s) > 1 else s[0]
    return t0 + s


def kappa(worldline, t_meas, transfer):
    """Pair clock of Eq. (10): sqrt(1-v^2)(t_meas - t0), t0 the cone crossing."""
    x0, t0, v = worldline
    xT, tT = transfer
    tc = crossing_time(x0, t0, v, xT, tT)
    v = np.asarray(v, float)
    return float(np.sqrt(1.0 - v @ v) * (t_meas - tc)), tc


def lorentz_event(L, t, x):
    e = L @ np.concatenate([[t], np.asarray(x, float)])
    return e[0], e[1:]


def transform_worldline(L, worldline):
    x0, t0, v = worldline
    t0p, x0p = lorentz_event(L, t0, x0)
    t1p, x1p = lorentz_event(L, t0 + 1.0, np.asarray(x0, float) + np.asarray(v, float))
    return (x0p, t0p, (x1p - x0p) / (t1p - t0p))


# ----------------------------------------------------------------- photons
def photon_affine_order_flips(tA=1.0, tB=1.3, betas=np.linspace(-0.9, 0.9, 181)):
    """Count boosts under which the coordinate/affine-time order of two
    back-to-back photons (along -x and +x from the origin) flips."""
    flips = 0
    for beta in betas:
        L = boost(np.array([beta, 0.0, 0.0]))
        tAp, _ = lorentz_event(L, tA, np.array([-tA, 0.0, 0.0]))
        tBp, _ = lorentz_event(L, tB, np.array([tB, 0.0, 0.0]))
        if (tAp < tBp) != (tA < tB):
            flips += 1
    return flips


def massive_proper_order_flips(tA=1.0, tB=1.3, speed=0.5, betas=np.linspace(-0.9, 0.9, 181)):
    """Same scan for a massive pair at +-speed: proper-time order never flips."""
    vA = np.array([-speed, 0.0, 0.0])
    vB = np.array([speed, 0.0, 0.0])
    flips = 0
    for beta in betas:
        L = boost(np.array([beta, 0.0, 0.0]))
        wlAp = transform_worldline(L, (np.zeros(3), 0.0, vA))
        wlBp = transform_worldline(L, (np.zeros(3), 0.0, vB))
        tAp, _ = lorentz_event(L, tA, vA * tA)
        tBp, _ = lorentz_event(L, tB, vB * tB)
        tauA = np.sqrt(1 - wlAp[2] @ wlAp[2]) * (tAp - wlAp[1])
        tauB = np.sqrt(1 - wlBp[2] @ wlBp[2]) * (tBp - wlBp[1])
        if (tauA < tauB) != (tA < tB):
            flips += 1
    return flips


# ------------------------------------------- P2 normalization singleton bound
def negativity(rho, dA, dB):
    """Negativity across the (dA | dB) cut of a bipartite density matrix."""
    r = rho.reshape(dA, dB, dA, dB).transpose(0, 3, 2, 1).reshape(dA * dB, dA * dB)
    ev = np.linalg.eigvalsh(r)
    return float(-ev[ev < 0].sum())
