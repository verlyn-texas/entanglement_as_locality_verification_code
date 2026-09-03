"""R23 — the anti-"distance" programme (Tier C3-C5): Propositions 5-7."""
import math
import unittest

import numpy as np

from mapping_spaces.entangled import qm
from mapping_spaces.entangled import r23_anti_distance as r23

NA = {"a": np.array([0.0, 0.0, 1.0]), "a'": np.array([1.0, 0.0, 0.0])}
NB = {"b": np.array([1.0, 0.0, 1.0]) / math.sqrt(2),
      "b'": np.array([1.0, 0.0, -1.0]) / math.sqrt(2)}


class TestC3LocalGenerator(unittest.TestCase):
    """Proposition 5: implementation locality."""

    def test_support_on_A_only(self):
        for n in (NA["a"], NA["a'"], np.array([0.6, 0.0, 0.8])):
            self.assertTrue(r23.support_is_A_only(n))

    def test_pushforward_equals_joint_projection_on_singlet(self):
        for n in (NA["a"], NA["a'"], np.array([0.6, 0.0, 0.8])):
            self.assertLess(r23.pushforward_is_projection(n), 1e-12)

    def test_jump_process_reproduces_quantum_box(self):
        E = r23.simulate_box(NA, NB, shots=200000, seed=7)
        S = E[("a", "b")] - E[("a", "b'")] + E[("a'", "b")] + E[("a'", "b'")]
        # each E has standard error ~1/sqrt(50000) ~ 0.0045; S ~ 0.009
        self.assertLess(abs(abs(S) - 2 * math.sqrt(2)), 0.05)
        for (sa, sb), e in E.items():
            na, nb = NA[sa], NB[sb]
            self.assertLess(abs(e - (-float(na @ nb))), 0.02)

    def test_law_is_quantum_tuned(self):
        # the jump probabilities ARE the Born weights: the generator imports
        # the quantum conditionals (the explanatory no-go via Proposition 2)
        rho = r23.singlet_dm()
        Ks = r23.measurement_kraus(NA["a"])
        ps = [float(np.real(np.trace(K @ rho @ K.conj().T))) for K in Ks]
        self.assertAlmostEqual(ps[0], 0.5, places=12)
        self.assertAlmostEqual(sum(ps), 1.0, places=12)


class TestC4OrderIndependence(unittest.TestCase):
    """Proposition 6: contact rules + order independence = no-signalling."""

    def test_one_way_box_excluded(self):
        P = r23.one_way_signalling_box()
        self.assertTrue(r23.realisable_A_first(P))
        self.assertFalse(r23.realisable_B_first(P))
        self.assertFalse(r23.order_independent_realisable(P))

    def test_pr_box_survives(self):
        P = r23.pr_box()
        self.assertTrue(r23.order_independent_realisable(P))
        self.assertTrue(r23.no_signalling(P))

    def test_singlet_box_survives(self):
        P = r23.singlet_box(NA, NB)
        self.assertTrue(r23.order_independent_realisable(P))

    def test_equivalence_with_no_signalling_on_random_boxes(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            P = rng.random((2, 2, 2, 2))
            P /= P.sum(axis=(0, 1), keepdims=True)
            self.assertEqual(r23.order_independent_realisable(P, 1e-6),
                             r23.no_signalling(P, 1e-6))


class TestC5QRFNoGo(unittest.TestCase):
    """Proposition 7: the chart is non-affine; any fixed unitary is affine."""

    def test_chart_velocity_non_affine(self):
        # Werner family: rho(p) = p singlet + (1-p) I/4 is itself affine in p,
        # so non-affinity of the chart velocity along it is non-affinity in rho
        f = r23.werner_chart_velocity
        # kink at p = 1/3: secant test around it
        lhs = f(1.0 / 3.0)
        sec = 0.5 * (f(1.0 / 6.0) + f(0.5))
        self.assertGreater(abs(lhs - sec), 0.05)
        defect = r23.affinity_defect(r23.chart_velocity_partner,
                                     r23.singlet_dm(), np.eye(4) / 4.0)
        self.assertGreater(defect, 0.05)

    def test_fixed_unitary_is_affine(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        V, _ = np.linalg.qr(x)
        O = np.diag([1.0, -1.0, 2.0, 0.5])
        g = r23.fixed_unitary_expectation(V, O)
        defect = r23.affinity_defect(g, r23.singlet_dm(), np.eye(4) / 4.0)
        self.assertLess(defect, 1e-10)
        # and for a second, random pair of states
        y = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho2 = y @ y.conj().T
        rho2 /= np.trace(rho2).real
        self.assertLess(r23.affinity_defect(g, rho2, np.eye(4) / 4.0), 1e-10)


if __name__ == "__main__":
    unittest.main()
