# Verification code for "A label-coordinate bookkeeping for entangled pairs: what co-location can and cannot explain"

This repository is the public code release referenced by the paper's "Code and
data availability" statement. Every "verified numerically" in the paper is
backed by a test here, and every figure and table of the paper is generated
from the same modules the tests check. Licence: MIT (see `LICENSE`).
Release `v1.1.0` is the version reviewed with the paper.

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

**Evaluation modules** (`mapping_spaces/entangled/`), each with a test file
`tests/test_entangled_<module>.py` of the same number:

| paper claim | section | module(s) |
|---|---|---|
| Prop. 1, frame maps, CHSH invariance, tracking, EPR positions | 3.1, 3.5, App. A–B | `r01_baseline`, `r07_realistic_tracking`, `r08_epr_positions` |
| Props. 2, 3, 6; the 1024/128/256 enumeration; PR box as contact rule | 3.2, 4.1 | `r02_contact_gate`, `r23_anti_distance` |
| persistent influence, second-round correlations, signalling 0.5, trial counts N(F) | 3.3(a) | `r06_qd_at_projection` |
| physical dimension: Kaluza–Klein tower, spreading time, Yukawa numbers | 3.3(b) | `r04_physical_qd`, `r15_qd_dimension` |
| locality fraction = weight, the Werner and pure families, 2N ≤ C | 3.4 | `r05_continuous_qd`, `r21_edge_weights` |
| locality graph, GHZ, swapping, mixed-state thresholds, Prop. 4, w ∈ [0, 1] | 3.6 | `graph_frames`, `r11_locality_graph`, `r12_swapping`, `r13_ghz`, `r17_mixed_multipartite`, `r21_edge_weights` |
| boost potentials, proper-time order, (R−1)/2R, non-relativistic limit | 3.7.1 | `lorentz_frames`, `r14_lorentz_frames` |
| the interval clock of P4, Prop. 5(a), the delayed-choice clocks | 3.7.2 | `r25_ordering_rule` |
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
