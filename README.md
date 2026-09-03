# Verification code for the paper — release manifest

This directory defines the public code release referenced by the paper's
"Code and data availability" statement ("Entanglement as locality in the
particles' own frames: a label-coordinate interpretation and its costs").
The code itself lives in the repository (single source of truth); this
manifest lists exactly what to publish and how a reader runs it. To assemble
a standalone release, copy the listed files preserving paths, plus this
README as the release README.

## Contents

**Shared core** (`mapping_spaces/`):
- `transformation.py` — the z-dependent shear frame maps (stdlib only)
- `entangled/qm.py`, `entangled/frames.py`, `entangled/graph_frames.py`,
  `entangled/lorentz_frames.py` — states/CHSH, label-coordinate frames,
  N-particle locality graph, boost-based frames

**Evaluation modules** — one behind each numbered claim of the paper:
- `entangled/r01_baseline.py` … `r17_mixed_multipartite.py` — the evaluations
  (Proposition 1 checks, contact-rule enumeration = Propositions 2/6, degree
  of locality, tracking, EPR positions, locality graph, GHZ, swapping,
  covariant variant, mixed multipartite)
- `entangled/r09_vs_collapse_models.py` — CSL/DP rates and geometry factors
  behind Sec. 3.10 (free-electron blindness, spin blindness, the zero
  prediction)
- `entangled/r18_cps_proposal.py` — the Cooper-pair-splitter numbers of
  Sec. 4.7 (S = 2.715, 422 coincidences, the 0.841 threshold) and Fig. 5
- `entangled/r21_edge_weights.py` — the continuous per-edge weight rule
  (Proposition 3, the repaired spectator counterexample)
- `entangled/r22_retarded_updates.py` — retarded graph updates
  (Proposition 4: chart locality; delayed-choice geometry)
- `entangled/r23_anti_distance.py` — the anti-"distance" programme
  (Propositions 5–7: chart-local generator, order-independence narrowing,
  QRF no-go)
- `entangled/r24_dim3.py` — the charts in 3+1 dimensions (Appendix C:
  vector Galilean form, group-valued covariant potentials with
  right-quotient composition, Wigner clause, Eq. (10) in vector form, the
  no-photonic-ordering scan, the singleton-cut bound behind w ∈ [0, 1])

**Tests** (`tests/`): the matching `test_entangled_r<nn>.py` files for the
modules above (r01–r18, r21–r24) plus `test_entangled_graph_frames.py` —
every numbered claim in the paper has a test; the suite is run with

    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python -m unittest discover -s tests -v

**Figures**: `solutions/entangled_particles/paper/figures.py` regenerates
every figure in the paper from the same modules the tests check (the paper
uses `fig_frames`, `fig_contact_polytope`, `fig_locality_graph`,
`fig_ordering`, `fig_cps`, `fig_tracking`). When assembling the release,
drop the figure functions not in that list together with the
`r19_force_noise_proposal` import they depend on — that module is not part
of this release.

## Status

The release is assembled but not yet public (GitHub issue #2 tracks
publication). Once a repository URL exists, replace "available from the
author on request; a public repository is in preparation" in the paper's
availability statement with the URL.
