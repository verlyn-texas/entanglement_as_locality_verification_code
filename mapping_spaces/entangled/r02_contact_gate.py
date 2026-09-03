"""R02 — the gate: what a *classical local rule at the contact point* in the
particles' frame can and cannot do.

In the particles' frames both electrons are at x = 0 for all lab times y, and
y is absolute, so the measurement made first (in y) is first in every frame.
A "contact model" lets A answer first, a = f(x), and lets B's answer depend
on everything present at the contact point: b = g(x, a, y).

Claims (numbered as in rows/R02_contact_gate.md):
 1. Deterministic contact strategies reach |S| = 4, but every strategy with
    |S| > 2 signals (B's marginal depends on A's setting).
 2. The no-signalling deterministic strategies are exactly the 256 local
    deterministic strategies (|S| <= 2).
 3. Stochastic contact strategies reach the PR box: S = 4 with no signalling
    -> the contact model spans the whole no-signalling polytope.
 4. The contact model tuned to the QM conditionals reproduces the singlet:
    S = 2 sqrt 2 with no signalling; it is empirically identical to M1.
 5. Hence co-location does not *explain* the Tsirelson bound; it is imported.
"""
from __future__ import annotations

import numpy as np

from mapping_spaces.entangled import qm


def enumerate_deterministic():
    """Return arrays (S, signalling) over the 1024 deterministic strategies."""
    S, sig = [], []
    for _, _, box in qm.deterministic_contact_boxes():
        S.append(qm.box_chsh(box))
        sig.append(qm.signalling(box))
    return np.array(S), np.array(sig)


def pr_box_as_contact():
    """The PR box written as a stochastic contact rule: A answers uniformly at
    random, B copies A except for the (x,y) = (0,1) pair."""
    pa = np.full((2, 2), 0.5)
    pb = np.zeros((2, 2, 2, 2))
    for x in range(2):
        for ia in range(2):
            for y in range(2):
                pb[x, ia, y, ia ^ (1 if (x, y) == (0, 1) else 0)] = 1.0
    return qm.contact_box(pa, pb)


def qm_tuned_contact():
    a, ap, b, bp = qm.chsh_optimal_settings()
    return qm.qm_conditional_contact(qm.singlet(), (a, ap), (b, bp))


def summary() -> dict:
    S, sig = enumerate_deterministic()
    ns = sig < 1e-12
    pr = pr_box_as_contact()
    qc = qm_tuned_contact()
    return {
        "n_strategies": int(S.size),
        "max_abs_S": float(np.abs(S).max()),
        "n_with_S4": int((np.abs(S) > 4 - 1e-9).sum()),
        "all_S_above_2_signal": bool(np.all(sig[np.abs(S) > 2 + 1e-9] > 1e-12)),
        "n_no_signalling": int(ns.sum()),
        "max_abs_S_no_signalling": float(np.abs(S[ns]).max()),
        "pr_S": float(qm.box_chsh(pr)),
        "pr_signalling": float(qm.signalling(pr)),
        "qm_tuned_S": float(qm.box_chsh(qc)),
        "qm_tuned_signalling": float(qm.signalling(qc)),
    }


if __name__ == "__main__":
    for k, v in summary().items():
        print(f"{k:28s} {v}")
