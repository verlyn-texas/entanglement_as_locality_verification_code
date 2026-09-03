"""R21 — continuous per-edge locality weights (paper-1 revision, B3/D6)."""
import unittest

import numpy as np

from mapping_spaces.entangled import qm
from mapping_spaces.entangled import r17_mixed_multipartite as r17
from mapping_spaces.entangled import r21_edge_weights as r21


class TestSpectatorCounterexample(unittest.TestCase):
    """The D6 discontinuity is repaired."""

    def test_old_rule_jumps(self):
        # the pre-revision component weight -> 0 as eps -> 0 although the
        # 1-2 marginal approaches a pure singlet (the verified counterexample)
        self.assertAlmostEqual(r21.old_component_weight(0.01), 0.02, places=3)

    def test_new_rule_continuous(self):
        for eps in (0.3, 0.1, 0.03, 0.01):
            w = r21.spectator_weights(eps)
            # the singlet edge stays near 1 and approaches it as eps -> 0
            self.assertGreater(w[(0, 1)], 2 * 0.49 * (1 - eps**2))
            # the spectator edges vanish smoothly (~2 eps at small eps)
            self.assertLess(w[(0, 2)], 3 * eps)
            self.assertLess(w[(1, 2)], 3 * eps)
        w0 = r21.spectator_weights(0.0)
        self.assertAlmostEqual(w0[(0, 1)], 1.0, places=9)
        self.assertAlmostEqual(w0[(0, 2)], 0.0, places=9)
        # continuity at the split point: w_12(eps) -> w_12(0)
        self.assertLess(abs(r21.spectator_weights(1e-4)[(0, 1)] - 1.0), 1e-6)


class TestPairReduction(unittest.TestCase):
    """On a lone pair the rule is 2N, equal to C on the quantified families."""

    def test_werner_family(self):
        for p in (0.4, 0.5, 0.8, 0.9, 1.0):
            rho = qm.werner(p)
            w = r21.pair_weight_two_qubits(rho)
            C = qm.concurrence(rho)
            self.assertAlmostEqual(w, C, places=9)

    def test_pure_family(self):
        for th in (np.pi / 4, 0.3, 0.1):
            psi = np.zeros(4, complex)
            psi[1], psi[2] = np.cos(th), -np.sin(th)
            rho = np.outer(psi, psi.conj())
            self.assertAlmostEqual(r21.pair_weight_two_qubits(rho),
                                   qm.concurrence(rho), places=7)

    def test_below_concurrence_in_general(self):
        rng = np.random.default_rng(11)
        for _ in range(40):
            x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
            rho = x @ x.conj().T
            rho /= np.trace(rho).real
            w = r21.pair_weight_two_qubits(rho)
            C = qm.concurrence(rho)
            self.assertLessEqual(w, C + 1e-9)
            self.assertGreaterEqual(w, -1e-12)


class TestNormalisedRuleBeyondQubits(unittest.TestCase):
    """Round-2 R4 repair: w = 2 N_min / (d_min - 1) stays in [0, 1] for any
    local dimensions and reproduces the qubit rule when all dims are 2."""

    def test_qutrit_counterexample_repaired(self):
        # the refuted pre-revision value: 2N = 2 for a spin-1 pair ...
        self.assertAlmostEqual(r21.unnormalised_pair_weight(3), 2.0, places=9)
        # ... the normalised rule gives 1, so P1 pins the partner at rest
        rho = r21.max_entangled_pair(3)
        w = r21.edge_weight_dims(rho, 0, 1, (3, 3))
        self.assertAlmostEqual(w, 1.0, places=9)
        self.assertAlmostEqual((1 - w), 0.0, places=9)  # velocity fraction

    def test_maximally_entangled_any_dim(self):
        for d in (2, 3, 4, 5):
            w = r21.edge_weight_dims(r21.max_entangled_pair(d), 0, 1, (d, d))
            self.assertAlmostEqual(w, 1.0, places=9)

    def test_bounded_on_random_qutrit_states(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            k = int(rng.integers(1, 10))
            x = rng.normal(size=(9, k)) + 1j * rng.normal(size=(9, k))
            rho = x @ x.conj().T
            rho /= np.trace(rho).real
            w = r21.edge_weight_dims(rho, 0, 1, (3, 3))
            self.assertGreaterEqual(w, -1e-12)
            self.assertLessEqual(w, 1.0 + 1e-9)

    def test_mixed_dimensions_bounded(self):
        # qubit (x) qutrit: d_min = 2, N <= 1/2 across the pair cut
        rng = np.random.default_rng(9)
        for _ in range(20):
            x = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
            rho = x @ x.conj().T
            rho /= np.trace(rho).real
            w = r21.edge_weight_dims(rho, 0, 1, (2, 3))
            self.assertGreaterEqual(w, -1e-12)
            self.assertLessEqual(w, 1.0 + 1e-9)

    def test_reduces_to_qubit_rule(self):
        g = r17.ghz(3)
        rho = np.outer(g, g.conj())
        for i, j in ((0, 1), (0, 2), (1, 2)):
            self.assertAlmostEqual(r21.edge_weight_dims(rho, i, j, (2, 2, 2)),
                                   r21.edge_weight(rho, i, j, 3), places=12)
        rho_w = r17.white_noise(r17.ghz(3), 0.6)
        self.assertAlmostEqual(r21.edge_weight_dims(rho_w, 0, 1, (2, 2, 2)),
                               r21.edge_weight(rho_w, 0, 1, 3), places=12)

    def test_isotropic_qutrit_continuity(self):
        # w is continuous in the state along the isotropic family
        prev = None
        for p in np.linspace(0.0, 1.0, 21):
            rho = p * r21.max_entangled_pair(3) + (1 - p) * np.eye(9) / 9
            w = r21.edge_weight_dims(rho, 0, 1, (3, 3))
            self.assertGreaterEqual(w, -1e-12)
            self.assertLessEqual(w, 1.0 + 1e-9)
            if prev is not None:
                self.assertLess(abs(w - prev), 0.11)
            prev = w
        self.assertAlmostEqual(prev, 1.0, places=9)


class TestMultipartite(unittest.TestCase):
    def test_ghz_edges_full(self):
        for n in (3, 4):
            g = r17.ghz(n)
            rho = np.outer(g, g.conj())
            for (i, j), w in r21.all_edge_weights(rho, n).items():
                self.assertAlmostEqual(w, 1.0, places=6)

    def test_ghz_measurement_z_kills_far_edge(self):
        # measuring particle 0 of GHZ_3 along Z leaves |000> or |111>
        rho = np.zeros((8, 8), complex)
        rho[0, 0] = 1.0
        self.assertAlmostEqual(r21.edge_weight(rho, 1, 2, 3), 0.0, places=9)

    def test_ghz_measurement_x_keeps_far_edge(self):
        psi = np.zeros(8, complex)
        psi[0b000] = psi[0b011] = 1 / np.sqrt(2)   # particle 0 in |+>, 1-2 Bell
        rho = np.outer(psi, psi.conj())
        self.assertAlmostEqual(r21.edge_weight(rho, 1, 2, 3), 1.0, places=6)

    def test_noisy_ghz_threshold_unchanged(self):
        # w > 0 exactly above the old component threshold 1/(1+2^(n-1))
        for n in (3, 4):
            thr = 1.0 / (1.0 + 2.0 ** (n - 1))
            for p, positive in ((thr * 0.95, False), (thr * 1.05, True)):
                rho = r17.white_noise(r17.ghz(n), p)
                w = r21.edge_weight(rho, 0, 1, n)
                self.assertEqual(w > 1e-9, positive, (n, p))

    def test_continuity_of_pinned_potentials(self):
        # the pinned value (1 - w)(vj - vi) inherits continuity: small state
        # perturbations move every weight by a small amount
        rng = np.random.default_rng(3)
        rho = r17.white_noise(r17.ghz(3), 0.7)
        w0 = r21.all_edge_weights(rho, 3)
        x = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        pert = x @ x.conj().T
        pert /= np.trace(pert).real
        for delta in (1e-3, 1e-5):
            rho_d = (1 - delta) * rho + delta * pert
            wd = r21.all_edge_weights(rho_d, 3)
            for k in w0:
                self.assertLess(abs(wd[k] - w0[k]), 20 * delta)


if __name__ == "__main__":
    unittest.main()
