"""Figures for the paper, generated from the row modules (paper conventions:
t for lab time, v for the particle speed = 1/a).  Run:

    .venv/bin/python solutions/entangled_particles/paper/figures.py

Writes PNG (300 dpi) and SVG into solutions/entangled_particles/paper/figures/.
Every number plotted is produced by the same code the row tests check.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mapping_spaces.entangled import qm, frames
from mapping_spaces.entangled import r02_contact_gate as r02
from mapping_spaces.entangled import r14_lorentz_frames as r14
from mapping_spaces.entangled import r18_cps_proposal as r18

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 120})
TS = 2 * np.sqrt(2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)


# 1. Worldlines in the lab frame and in a particle's frame (v = 1/a)
def fig_frames(v=0.5):
    a = 1 / v
    t = np.linspace(0, 4, 50)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    xA, xB = frames.lab_tracks(a, t)
    axes[0].plot(xA, t, label="electron A", color="C0")
    axes[0].plot(xB, t, label="electron B", color="C3")
    axes[0].plot(0 * t, t, "k--", lw=1, label="laboratory")
    axes[0].set_title("Laboratory frame ($z=0$)")
    axes[0].set_xlabel("$x$ (arbitrary units)")
    axes[0].set_ylabel("$t$ (arbitrary units)")
    axes[0].legend(loc="upper left", fontsize=7)
    # A's frame: A and B at x=0; lab at x = v t (sigma_A(0) = v)
    axes[1].plot(0 * t, t, color="C0", lw=3, alpha=0.5, label="A and B (co-located)")
    axes[1].plot(0 * t, t, color="C3", lw=1)
    lab = frames.path_in_frame(frames.LAB, frames.A, a, lambda s: 0 * s, t) if False else v * t
    axes[1].plot(lab, t, "k--", lw=1, label="laboratory")
    axes[1].set_title("Electron A's frame ($z=-1$)")
    axes[1].set_xlabel("$x'$ (arbitrary units)")
    axes[1].legend(loc="upper left", fontsize=7)
    for ax in axes:
        ax.set_xlim(-2.2, 2.2)
        ax.grid(alpha=0.3)
    save(fig, "fig_frames")


# 2. The gate: |S| over deterministic contact strategies
def fig_contact_polytope():
    S, sig = r02.enumerate_deterministic()
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    bins = np.arange(-4.25, 4.5, 0.5)
    ax.hist([S[sig < 1e-12], S[sig > 1e-12]], bins=bins, stacked=True, color=["C0", "C3"],
            label=["no-signaling (local-realist) strategies", "signaling strategies"])
    for val, lab, ls in ((2, "local-realist bound 2", ":"), (TS, r"Tsirelson bound $2\sqrt{2}$", "-"), (4, "PR box 4", "--")):
        ax.axvline(val, color="k", ls=ls, lw=1, label=lab)
    ax.set_ylim(0, 470)
    ax.set_xlabel("CHSH value $S$ of a deterministic contact strategy")
    ax.set_ylabel("count (of 1024)")
    ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig_contact_polytope")


# 4. Tracking precision (R07 claim 4 closed form) vs measurement strength
def fig_tracking(m=9.1093837e-31, hbar=1.054571817e-34, v0=1e4):
    ks = np.logspace(18, 27, 60)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for T, c in ((1e-8, "C0"), (1e-7, "C1"), (1e-6, "C2")):
        gam = np.sqrt(8 * ks * hbar / m)
        var = (2 * hbar / m) * (1.2 / T + gam / 2) + (52 / 35) * hbar ** 2 * ks * T / m ** 2
        ax.loglog(ks, np.sqrt(var) / (2 * v0), color=c, label=f"run time {T*1e9:.0f} ns")
    ax2 = ax.twinx()
    ax2.loglog(ks, np.sqrt(np.sqrt(hbar / (8 * m * ks))) * 1e6, "k--", lw=1)
    ax2.set_ylabel("steady-state $\\sqrt{V_x}$ (μm, dashed)")
    ax.set_xlabel("measurement strength $\\Gamma_m$ (m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel("relative error of the separation speed $2v$")
    ax.axhline(0.01, color="gray", ls=":", lw=1)
    ax.legend(fontsize=7, loc="center left")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_tracking")


# 5. Cooper-pair-splitter proposal: S_obs vs filter polarization and vs transfer fidelity
def fig_cps():
    # Single panel (round-3 revision, A7): the polarized-dot-filter panel only.
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    P = np.linspace(0.5, 1, 101)
    for eta, c in ((1.0, "C0"), (0.85, "C1")):
        ax.plot(P, [r18.s_obs_filters(1.0, eta, x, x) for x in P], color=c, label=f"$\\eta={eta}$")
    ax.axhline(2, color="k", ls=":", lw=1)
    ax.axvline(0.60, color="C3", ls="--", lw=1)
    ax.text(0.61, 0.3, "nanowire filters\n$\\bar P\\approx0.60$", fontsize=7, color="C3")
    ax.axvline(np.sqrt(0.96), color="C2", ls="--", lw=1)
    ax.text(np.sqrt(0.96) - 0.01, 0.3, "2DEG\n$E_{\\rm cc}=-0.96$", fontsize=7, color="C2", ha="right")
    ax.set_xlabel("spin-filter polarization $P$ (symmetric)")
    ax.set_ylabel("$S_{\\rm obs}$")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(0, 3)
    ax.grid(alpha=0.3)
    save(fig, "fig_cps")


# 5b. The locality graph at the three kinds of event (schematic, three panels)
def fig_locality_graph():
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))

    def node(ax, x, y, label, lab_node=False):
        ax.scatter([x], [y], s=420 if lab_node else 150,
                   c="lightgray" if lab_node else "C0",
                   edgecolors="k", zorder=3)
        ax.annotate(label, (x, y), ha="center", va="center", fontsize=10, zorder=4)

    def edge(ax, p, q, w=1.0, style="-"):
        ax.plot([p[0], q[0]], [p[1], q[1]], style, color="C3",
                lw=2.6 * w + 0.4, alpha=0.9, zorder=2)

    # panel 1: creation — a co-located entangling event makes an edge
    ax = axes[0]
    ax.set_title("entangling event\n(edge created, $w=1$)", fontsize=10)
    node(ax, 0.5, 0.72, "lab", lab_node=True)
    node(ax, 0.3, 0.35, "A")
    node(ax, 0.7, 0.35, "B")
    edge(ax, (0.3, 0.35), (0.7, 0.35))
    ax.text(0.5, 0.50, "created together\nat one lab event", fontsize=9, ha="center")

    # panel 2: measurement — the component splits, edges vanish
    ax = axes[1]
    ax.set_title("first measurement on A\n(edges follow the components)", fontsize=10)
    node(ax, 0.5, 0.72, "lab", lab_node=True)
    node(ax, 0.18, 0.35, "A")
    node(ax, 0.46, 0.35, "B")
    edge(ax, (0.18, 0.35), (0.46, 0.35), w=0.0, style=":")
    node(ax, 0.72, 0.35, "C")
    node(ax, 0.94, 0.35, "D")
    edge(ax, (0.72, 0.35), (0.94, 0.35), w=0.7)
    ax.text(0.20, 0.16, "projected pair:\n$w \\to 0$", fontsize=9)
    ax.text(0.70, 0.16, "partial ent.:\n$w = 2\\mathcal{N}$", fontsize=9)

    # panel 3: Bell-state measurement — the transfer postulate
    ax = axes[2]
    ax.set_title("Bell-state measurement on B, C\n(transfer: edge $A$–$D$)", fontsize=10)
    node(ax, 0.5, 0.72, "lab", lab_node=True)
    node(ax, 0.12, 0.42, "A")
    node(ax, 0.38, 0.42, "B")
    node(ax, 0.62, 0.42, "C")
    node(ax, 0.88, 0.42, "D")
    edge(ax, (0.12, 0.42), (0.38, 0.42), w=0.0, style=":")
    edge(ax, (0.62, 0.42), (0.88, 0.42), w=0.0, style=":")
    ax.annotate("BSM", (0.5, 0.50), ha="center", fontsize=9)
    ax.plot([0.5], [0.42], marker="x", ms=9, color="k", zorder=4)
    ax.add_patch(matplotlib.patches.FancyArrowPatch(
        (0.12, 0.47), (0.88, 0.47), connectionstyle="arc3,rad=0.35",
        arrowstyle="-", color="C3", lw=2.6, zorder=2))
    ax.text(0.5, 0.08, "old edges removed (dotted); $A$–$D$ created\n(covariantly: on the future cone of the BSM)",
            fontsize=9, ha="center")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    save(fig, "fig_locality_graph")


# 8. Fraction of configurations where proper-time order differs from lab order
def fig_ordering():
    vB = np.linspace(0, 0.95, 60)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for vA, c in ((-0.1, "C0"), (-0.3, "C1"), (-0.6, "C2")):
        ax.plot(vB, [r14.disagreement_fraction_closed_form(vA, x) for x in vB], color=c, label=f"$v_A={vA}c$")
    ax.set_xlabel("$v_B$ (units of $c$)")
    ax.set_ylabel("fraction of $(t_A,t_B)$ with different order")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, "fig_ordering")


if __name__ == "__main__":
    for f in (fig_frames, fig_contact_polytope, fig_tracking, fig_cps,
              fig_locality_graph, fig_ordering):
        f()
        print("wrote", f.__name__)
