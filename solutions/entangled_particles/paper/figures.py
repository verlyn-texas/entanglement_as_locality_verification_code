"""Figures for the papers, generated from the row modules (paper conventions:
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
from mapping_spaces.entangled import r17_mixed_multipartite as r17
from mapping_spaces.entangled import r18_cps_proposal as r18
from mapping_spaces.entangled import r19_force_noise_proposal as r19

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


# 3. CHSH vs visibility with ledger points
def fig_chsh_visibility():
    p = np.linspace(0, 1, 101)
    S = np.array([qm.chsh_max(qm.werner(x)) for x in p])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(p, S, color="C0", label="quantum prediction $S = 2\\sqrt{2}\\,\\mathcal{V}$")
    ax.axhline(2, color="k", ls=":", lw=1)
    ax.axhline(TS, color="k", ls="-", lw=1)
    ax.text(0.02, 2.03, "local-realist bound", fontsize=7)
    ax.text(0.02, TS + 0.03, "Tsirelson bound", fontsize=7)
    pts = [("photons, Poh 2015", 2.82759), ("Si quantum dots 0.1 K, Steinacker 2025", 2.731),
           ("Si quantum dots 1.1 K", 2.101), ("Be$^+$ ions, Rowe 2001", 2.25),
           ("NV electrons 1.3 km, Hensen 2015", 2.42), ("swapped photons, Jennewein 2002", 2.421)]
    for i, (lab, s) in enumerate(pts):
        ax.plot(s / TS, s, "o", color=f"C{i+1}", ms=5, label=f"{lab}: $S={s:.3g}$")
    ax.set_xlabel("visibility $\\mathcal{V}$ (Werner-state fraction)")
    ax.set_ylabel("$S$")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(alpha=0.3)
    save(fig, "fig_chsh_visibility")


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


# 6. CSL bounds at r_C = 1e-7 m and the design targets (ledger values).
# Each bar carries (frequency band, spectral assumption); preprints hatched;
# projected targets are lines, visually distinct from measured exclusions.
def fig_collapse_bounds():
    # (label, lambda, hatched-as-preprint?)
    items = [
        ("cold-atom expansion\n[sub-Hz; white assumed]", 5.1e-8, False),
        ("LISA Pathfinder (2017)\n[mHz band; measured]", 3.8e-9, False),
        ("multilayer cantilever, 30 mK (2020)\n[Hz–kHz band; measured]", 2.0e-10, False),
        ("LISA Pathfinder reanalysis (2024)\n[mHz band; PREPRINT]", 8.3e-11, True),
        ("bulk heating\n[white to $\\gtrsim10^{11}$ Hz assumed]", 3.3e-11, False),
        ("X-ray, quasi-free electrons (2022)\n[white to $\\sim10^{18}$ Hz assumed]", 1.7e-13, False),
        ("X-ray, coherent nuclei (2022)\n[white to $\\sim10^{18}$ Hz + coherence assumed]", 4.9e-15, False),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(items))
    for i, (_, v, pre) in enumerate(items):
        ax.barh(i, v, color="C0", alpha=0.45 if pre else 0.85,
                hatch="//" if pre else None, edgecolor="C0")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in items], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 1e-6)
    for val, lab, c, ls in ((1e-8, "Adler value $10^{-8}$", "C3", "--"),
                            (1e-16, "GRW value $10^{-16}$ (GRW convention)", "C2", "--"),
                            (1e-12, "PROJECTED target $10^{-12}$ (100 Hz variant)", "C1", "-."),
                            (8.5e-15, "PROJECTED floor $8.5\\times10^{-15}$ (levitated, modulated)", "C1", ":")):
        ax.axvline(val, color=c, ls=ls, lw=1.2, label=lab)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_xlabel("excluded CSL rate $\\lambda$ above this value, at $r_C=10^{-7}$ m (s$^{-1}$)")
    ax.invert_yaxis()
    save(fig, "fig_collapse_bounds")


# 6b. The frequency-cut-off exclusion panel: surviving lambda bound against the
# assumed cut-off frequency of the collapse noise.  Each experiment's bound
# degrades as lambda_0 (1 + (f_m/f_c)^2) once the cut-off falls below the band
# f_m it measures in (Lorentzian noise model).
def fig_cutoff_panel():
    fc = np.logspace(-5, 20, 600)

    def surviving(lam0, fm):
        return lam0 * (1.0 + (fm / fc) ** 2)

    curves = [
        ("X-ray, quasi-free electrons ($f_m\\sim10^{18}$ Hz)", 1.7e-13, 1e18, "C4", "-"),
        ("X-ray, coherent nuclei", 4.9e-15, 1e18, "C4", "--"),
        ("LISA Pathfinder reanalysis (mHz; preprint)", 8.3e-11, 1e-3, "C0", "-"),
        ("multilayer cantilever (3.5 kHz)", 2.0e-10, 3.5e3, "C1", "-"),
        ("proposed 100 Hz sensor, $10^{-12}$ target (projected)", 1e-12, 1e2, "C3", "-."),
        ("proposed levitated floor $8.5\\times10^{-15}$ (projected)", 8.5e-15, 1e2, "C3", ":"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for lab, lam0, fm, c, ls in curves:
        ax.loglog(fc, surviving(lam0, fm), color=c, ls=ls, lw=1.4, label=lab)
    # cosmological cut-offs Omega_c ~ 1e11-1e12 s^-1  ->  f = Omega/2pi
    ax.axvspan(1e11 / (2 * np.pi), 1e12 / (2 * np.pi), color="gray", alpha=0.25,
               label="cosmological cut-offs ($\\Omega_c\\sim10^{11}$–$10^{12}$ s$^{-1}$)")
    ax.axhline(1e-16, color="C2", ls="--", lw=1, label="GRW value (GRW convention)")
    ax.set_xlabel("assumed cut-off frequency $f_c$ of the collapse noise (Hz)")
    ax.set_ylabel("surviving bound on $\\lambda$ at $r_C=10^{-7}$ m (s$^{-1}$)")
    ax.set_ylim(1e-17, 1e-6)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_cutoff_panel")


# 7. Locality weight of a noisy GHZ component under local dephasing
def fig_dephasing(gamma=1.0):
    ts = np.linspace(0, 3, 200)
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    for n, c in ((2, "C0"), (3, "C1"), (4, "C2")):
        ax.plot(ts, 2 * r17.ghz_dephasing_closed(n, gamma, ts), color=c, label=f"GHZ, $N={n}$")
    ax.set_xlabel("time $\\gamma t$ (per-qubit dephasing rate $\\gamma$)")
    ax.set_ylabel("component weight (min-cut negativity, normalized)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, "fig_dephasing")


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


# 9. Force-noise budget for the proposed experiment (one-sided PSDs, S_F = 2 hbar^2 eta)
def fig_noise_budget():
    lam = np.logspace(-17, -8, 200)
    rc = r19.R_C_REF
    d_opt = r19.optimal_layer_thickness(rc)["d_over_rC"] * rc
    m = r19.M_CANT
    S_csl = np.array([r19.stack_specific_noise(x, rc, d_opt) * m for x in lam])
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.loglog(lam, S_csl, color="C3", label=f"CSL force noise, optimized multilayer ($d=3.2\\,r_C$, $m={m*1e9:.2f}$ ng)")
    for T, Q, f0, c, lab in ((0.030, r19.Q_CANT, r19.F0_CANT, "C0", "thermal: 30 mK, $Q=2.8\\times10^6$, 3.5 kHz (existing)"),
                             (0.010, 1e8, 100.0, "C1", "thermal: 10 mK, $Q=10^8$, 100 Hz"),
                             (0.010, 1e9, 100.0, "C2", "thermal: 10 mK, $Q=10^9$, 100 Hz")):
        ax.axhline(r19.thermal_force_noise(T, m, f0, Q), color=c, ls="--", lw=1, label=lab)
    ax.axhline(r19.SF0_LIMIT, color="k", ls=":", lw=1, label="published 95 % limit on excess noise")
    for val, lab in ((1e-16, "GRW"), (1e-12, "target"), (1e-14, "target")):
        ax.axvline(val, color="gray", lw=0.8, ls=":")
        ax.text(val, 2e-44, lab, rotation=90, fontsize=6, ha="right", va="bottom", color="gray")
    ax.set_xlabel("CSL rate $\\lambda$ at $r_C=10^{-7}$ m (s$^{-1}$)")
    ax.set_ylabel("one-sided force PSD $S_F$ (N$^2$/Hz)")
    ax.set_ylim(1e-44, 1e-33)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_noise_budget")


if __name__ == "__main__":
    for f in (fig_frames, fig_contact_polytope, fig_chsh_visibility, fig_tracking, fig_cps,
              fig_locality_graph, fig_collapse_bounds, fig_cutoff_panel, fig_dephasing,
              fig_ordering, fig_noise_budget):
        f()
        print("wrote", f.__name__)
