# Verification code for "A label-coordinate bookkeeping for entangled pairs: what co-location can and cannot explain"

This repository is the public code release referenced by the paper's "Code and
data availability" statement. Every "verified numerically" in the paper is
backed by a test here, and every figure of the paper, together with the
numerical tables the paper moved into this repository, is generated from the
same modules the tests check. Licence: MIT (see `LICENSE`).
Release `v1.2.0` is the version reviewed with the paper (`v1.1.0` was the
previous round's; the difference is the tests listed under "Round-5
additions" below).

## Running the tests

    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python -m unittest discover -s tests -v

(Python 3.11 or later; NumPy is the only dependency of the tests. Figures need
matplotlib: `.venv/bin/pip install matplotlib` and then
`.venv/bin/python solutions/entangled_particles/paper/figures.py`.)

## Layout

**Shared core** (`mapping_spaces/`):
- `transformation.py` — the z-dependent shear frame maps (stdlib only)
- `entangled/qm.py`, `entangled/frames.py`, `entangled/graph_frames.py`,
  `entangled/lorentz_frames.py` — states/CHSH, label-coordinate frames,
  N-particle locality graph, boost-based frames

**Evaluation modules** (`mapping_spaces/entangled/`), each numbered module
`rNN_*` with the test file `tests/test_entangled_rNN.py` of its number
(`graph_frames` has its own test file; `lorentz_frames` is exercised under
`r14`, `r16` and `graph_frames`):

| paper claim | section | module(s) |
|---|---|---|
| Prop. 1, frame maps, CHSH invariance, tracking, EPR positions | 3.1, 3.5, App. A–B | `r01_baseline`, `r07_realistic_tracking`, `r08_epr_positions` |
| Props. 2, 3, 6; the 1024/128/256 enumeration; PR box as contact rule | 3.2, 4.1 | `r02_contact_gate`, `r23_anti_distance` |
| persistent influence, second-round correlations, signalling 0.5, trial counts N(F) | 3.3(a) | `r06_qd_at_projection` |
| physical dimension: Kaluza–Klein tower, spreading time, Yukawa numbers | 3.3(b) | `r04_physical_qd`, `r15_qd_dimension` |
| locality fraction = weight, the Werner and pure families, 2N ≤ C | 3.4 | `r05_continuous_qd`, `r21_edge_weights` |
| locality graph, GHZ, swapping, mixed-state thresholds, Prop. 4, w ∈ [0, 1]; the "When" rule (conditional vs dephased weights) | 3.6, canonical box | `graph_frames`, `r11_locality_graph`, `r12_swapping`, `r13_ghz`, `r17_mixed_multipartite`, `r21_edge_weights` |
| boost potentials, proper-time order, (R−1)/2R, non-relativistic limit | 3.7.1 | `lorentz_frames`, `r14_lorentz_frames` |
| the interval clock of P4, Prop. 5(a), the delayed-choice clocks; the inherited anchor of a pruned component; the non-composition of component orders | 3.7.2 | `r25_ordering_rule` |
| retarded chart updates, Prop. 5(b), cone-crossing equivariance | 3.7.2 | `r16_covariant_swapping`, `r22_retarded_updates` |
| the charts in 3+1 dimensions, Wigner scan, photon clocks, singleton-cut bound | App. C | `r24_dim3` |
| moving-apparatus invariance | 3.8 | `r03_absolute_time` |
| collapse rates, free-electron and spin blindness | 3.9 | `r09_vs_collapse_models` |
| Cooper-pair-splitter numbers of avenue (ii) | 4.3 | `r18_cps_proposal` |

`r10_cooper_pair_splitter` is the earlier form of the splitter numbers and is
kept for its tests. The numbering gap (`r19`, `r20`) corresponds to modules
behind claims that are not made in this paper; they are not part of this
release.

**Figures**: `solutions/entangled_particles/paper/figures.py` regenerates the
paper's figures (`fig_frames`, `fig_locality_graph`, `fig_ordering`) and the
tables and figures the paper moved into this repository (the tracking table
and figure of Appendix A: `fig_tracking`; the free-flight table of Appendix B
is printed by `r08_epr_positions`; `fig_contact_polytope` and `fig_cps` are
the histogram of the contact-rule enumeration and the splitter plot).

## Round-5 additions (v1.2.0)

- `r12_swapping.dephased_vs_conditional_weights` — P2 weights on the dephased
  versus the outcome-conditioned state (single-particle projections agree; a
  Bell-basis projection gives w_AD = w_BC = 0 versus 1): the charts are read
  off the conditional state.
- `r25_ordering_rule.pruned_pair_anchors`, `random_spacelike_triples_pruning_anchor`
  — a pruned component inherits its parent's defining event; the pruning
  anchor would leave the remnant of a mutually spacelike GHZ triple unordered.
- `r25_ordering_rule.cross_component_cycle`, `cross_component_cycle_rate` —
  the component orders and the causal order can form a cycle (about 0.25 % of
  random swapping geometries, delayed-choice branch only).
- `r16_covariant_swapping.chart_jump_at_reanchoring` — the covariant chart
  jump in the working pair (B: 0 → 2.14, D: 4.13 → 1.85 in A's chart at t = 3.5).
- `r07_realistic_tracking.spin_step_closed_form` and the `position_state`
  option — |S| = √2 (2 − 0.2⟨x²⟩) after one Euler step; 2.47 on the uniform
  four-level grid, 2.53 for the seeded state (⟨x²⟩ = 1.066).
- `r24_dim3.wigner_scan_max`, `wigner_angle_two_boosts` (26.6° in the scan,
  34.44° for two perpendicular 0.85c boosts) and `r08_epr_positions.tms_negativity`
  ((e^{2r} − 1)/2 = 0.86, 3.19, 26.8) — the Appendix C numbers now asserted.
