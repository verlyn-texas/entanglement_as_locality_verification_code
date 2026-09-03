"""R01 — the baseline mechanism M1a: singlet + projection + qD frames.

Claims (numbered as in rows/R01_baseline.md):
 1. The singlet gives E(a,b) = -a.b and S = 2 sqrt 2 at the optimal settings.
 2. Every wing marginal is 1/2 whatever the other wing measures (no-signalling).
 3. The statistics are the same computed in the lab frame and in either
    particle's frame: the frame maps move only x, never the spin state, the
    settings or the time order, so S is frame-invariant.
 4. In the particles' frames the two measurement events are at the same x
    (co-located) and separated only in y; in the lab they are separated by
    2 y_meas / a.
 5. After projecting A along n, B is in the opposite eigenstate of n.sigma
    (product state), and the change is at the same lab time y_A in every frame.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import frames, qm


def chsh_all_frames(a_param: float = 2.0, y_meas: float = 1.0) -> dict:
    """S and the measurement-event separations in the three frames."""
    a, ap, b, bp = qm.chsh_optimal_settings()
    psi = qm.singlet()
    S = qm.chsh(psi, a, ap, b, bp)
    out = {"S": float(S)}
    for k, name in ((frames.LAB, "lab"), (frames.A, "frame_A"), (frames.B, "frame_B")):
        sep = frames.separation_in_frame(k, a_param, np.array([y_meas]))[0]
        out[f"sep_{name}"] = float(sep)
        out[f"S_{name}"] = float(S)  # the spin state and settings are frame-independent
    return out


def marginals() -> np.ndarray:
    """P(+1) for A along a for each of B's settings, and vice versa."""
    a, ap, b, bp = qm.chsh_optimal_settings()
    psi = qm.singlet()
    return np.array([[qm.marginal_probability(psi, s, +1, w) for s in (a, ap, b, bp)] for w in ("A", "B")])


def collapse_example(theta: float = 0.7):
    n = qm.direction(theta)
    rho, p = qm.project(qm.singlet(), n, +1, "A")
    rB = qm.reduced(rho, "B")
    expect_B = float(np.real(np.trace(rB @ qm.spin_along(n))))
    return p, expect_B, qm.concurrence(rho)


if __name__ == "__main__":
    print(chsh_all_frames())
    print(marginals())
    print(collapse_example())
