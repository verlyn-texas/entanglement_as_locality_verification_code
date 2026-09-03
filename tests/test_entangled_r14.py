"""Tests for row R14 (relativistic, 1+1 D, c = 1): every numbered claim of
solutions/entangled_particles/rows/R14_lorentz_frames.md."""
import unittest

import numpy as np

from mapping_spaces import transformation as T
from mapping_spaces.entangled import graph_frames as G, lorentz_frames as LZ, qm, r14_lorentz_frames as r

VA, VB = -0.3, 0.8  # asymmetric lab velocities used throughout
TA, TB = 1.0, 1.2   # lab times of the two measurement events


class R14(unittest.TestCase):
    def setUp(self):
        self.g = r.pair(VA, VB)
        self.h = r.with_third(self.g, "C", 0.4, 0.1)  # unrelated, w = 0

    # ------------------------------------------------------------ claim 1
    def test_claim1_composition_and_partner_at_rest(self):
        rng = np.random.default_rng(14)
        for _ in range(5):
            z, t, x = rng.uniform(-1, 1), rng.uniform(0, 2), rng.uniform(-1, 1)
            for k, l, m in (("A", "B", "C"), ("C", "A", G.LAB), (G.LAB, "B", "A")):
                self.assertLess(r.composition_defect(self.h, k, l, m, z, t, x), 1e-12)
        # pinning: psi_k(z_k) = psi_k(0) = artanh v_k, psi_k(z_j) = artanh v_j for w = 1
        self.assertAlmostEqual(r.rapidity_at(self.g, "A", -1.0), np.arctanh(VA), places=12)
        self.assertAlmostEqual(r.rapidity_at(self.g, "A", 0.0), np.arctanh(VA), places=12)
        self.assertAlmostEqual(r.rapidity_at(self.g, "A", 1.0), np.arctanh(VB), places=12)
        for g in (self.g, r.pair(-0.6, 0.6)):
            d = r.partner_rest(g, "A", "B", TA, TB)
            self.assertAlmostEqual(d["v_j_in_i"], 0.0, places=12)
            self.assertAlmostEqual(d["v_i_in_j"], 0.0, places=12)
            for xs in (d["x_in_i"], d["x_in_j"]):
                self.assertAlmostEqual(xs[0], 0.0, places=12)
                self.assertAlmostEqual(xs[1], 0.0, places=12)
        self.assertAlmostEqual(r.lab_velocity_in_frame(self.g, "A"), -VA, places=12)
        self.assertAlmostEqual(r.lab_velocity_in_frame(self.g, "B"), -VB, places=12)

    # ------------------------------------------------------------ claim 2
    def test_claim2_frame_times_are_proper_times_and_boost_invariant(self):
        tA_, tB_ = r.frame_times(self.g, "A", "A", "B", TA, TB)
        self.assertAlmostEqual(tA_, LZ.proper_time(VA, TA), places=12)  # 0.95394
        self.assertAlmostEqual(tB_, LZ.proper_time(VB, TB), places=12)  # 0.72
        self.assertAlmostEqual(tA_, 0.9539392, places=6)
        self.assertAlmostEqual(tB_, 0.72, places=12)
        # same numbers from B's frame
        np.testing.assert_allclose(r.frame_times(self.g, "B", "A", "B", TA, TB), (tA_, tB_), atol=1e-12)
        self.assertEqual(LZ.order_in_pair_frame(self.g, "A", "B", TA, TB), "B")
        for u in (-0.5, 0.2, 0.9):
            first, (tA_u, tB_u) = r.ordering_under_boost(self.g, u, "A", "B", TA, TB)
            self.assertEqual(first, "B")
            self.assertAlmostEqual(tA_u, tA_, places=12)
            self.assertAlmostEqual(tB_u, tB_, places=12)

    # ------------------------------------------------------------ claim 3
    def test_claim3_lab_vs_proper_time_ordering(self):
        # the example: A first in the lab, B first by proper time
        self.assertEqual(r.lab_order(TA, TB), "A")
        self.assertEqual(r.proper_order(TA, TB, VA, VB), "B")
        self.assertTrue(r.orderings_disagree(TA, TB, VA, VB))
        self.assertTrue(r.is_spacelike(TA, TB, VA, VB))
        # wedge boundary: disagreement iff 1 < t_B/t_A < gamma_B/gamma_A
        rr = r.disagreement_ratio_bound(VA, VB)
        self.assertAlmostEqual(rr, 1.5898987, places=6)
        self.assertTrue(r.orderings_disagree(1.0, 0.999 * rr, VA, VB))
        self.assertFalse(r.orderings_disagree(1.0, 1.001 * rr, VA, VB))
        self.assertFalse(r.orderings_disagree(1.0, 0.999, VA, VB))
        # closed-form fraction (r-1)/(2r) vs Monte Carlo
        cf = r.disagreement_fraction_closed_form(VA, VB)
        self.assertAlmostEqual(cf, 0.1855145, places=6)
        self.assertLess(abs(r.disagreement_fraction_mc(VA, VB) - cf), 0.01)
        # symmetric velocities never disagree
        for v in (0.1, 0.6, 0.9):
            self.assertEqual(r.disagreement_fraction_closed_form(-v, v), 0.0)
            self.assertEqual(r.disagreement_fraction_mc(-v, v), 0.0)
        # the causal order is always respected (disagreement only if spacelike)
        ok, n_causal, n_bad = r.causal_order_respected()
        self.assertTrue(ok)
        self.assertGreater(n_causal, 5000)
        self.assertEqual(n_bad, 0)

    # ------------------------------------------------------------ claim 4
    def test_claim4_pair_frame_is_not_a_global_lorentz_frame(self):
        # different rapidities at the two partners' heights
        self.assertNotAlmostEqual(r.rapidity_at(self.g, "A", -1.0), r.rapidity_at(self.g, "A", 1.0), places=3)
        # unrelated C: relativistic relative velocity, A's ordinary rest-frame time
        u = (0.1 - VA) / (1 - 0.1 * VA)
        self.assertAlmostEqual(LZ.velocity_in_frame(self.h, "C", "A"), u, places=12)  # 0.38835
        self.assertAlmostEqual(r.rapidity_at(self.h, "A", 0.4), np.arctanh(VA), places=12)
        tC = r.third_particle_frame_time(self.h, "A", "C", 1.0)
        self.assertAlmostEqual(tC, r.lorentz_rest_frame_time(VA, 0.1, 1.0), places=12)
        self.assertAlmostEqual(tC, 1.0797334, places=6)
        self.assertGreater(abs(tC - LZ.proper_time(0.1, 1.0)), 0.08)  # tau_C = 0.99499
        # partial weight interpolates the velocity linearly and the rapidity between A's and C's
        for w in (0.0, 0.5, 1.0):
            hw = r.with_third(self.g, "C", 0.4, 0.1, w)
            self.assertAlmostEqual(LZ.velocity_in_frame(hw, "C", "A"), (1 - w) * u, places=12)
        self.assertAlmostEqual(r.third_particle_frame_time(r.with_third(self.g, "C", 0.4, 0.1, 1.0), "A", "C", 1.0),
                               LZ.proper_time(0.1, 1.0), places=12)

    # ------------------------------------------------------------ claim 5
    def test_claim5_nonrelativistic_limit_scales_as_v_squared(self):
        sweep = r.nonrelativistic_sweep((1e-1, 1e-2, 1e-3))
        ratios = [sweep[e] / e**2 for e in (1e-1, 1e-2, 1e-3)]
        for q in ratios:
            self.assertAlmostEqual(q, 0.2344, places=3)
        self.assertAlmostEqual(sweep[1e-1] / sweep[1e-2], 100.0, delta=0.1)
        self.assertAlmostEqual(sweep[1e-2] / sweep[1e-3], 100.0, delta=0.1)
        # the lab potential is reproduced exactly (psi_L = 0)
        g = G.pass1_pair(a=10.0)
        z = np.linspace(-1, 1, 9)
        np.testing.assert_allclose(g.potential(G.LAB)(z) - np.tanh(LZ.rapidity_potential(g, G.LAB)(z)),
                                   T.shear_potential(3, 10.0)(z), atol=1e-15)

    # ------------------------------------------------------------ claim 6
    def test_claim6_statistics_are_order_independent(self):
        s = qm.chsh_optimal_settings()
        sA, sB = (s[0], s[1]), (s[2], s[3])
        qbox = qm.box_from_state(qm.singlet(), sA, sB)
        self.assertAlmostEqual(abs(qm.box_chsh(qbox)), qm.TSIRELSON, places=12)
        for first in ("A", "B"):
            box = r.sequential_box(qm.singlet(), sA, sB, first)
            np.testing.assert_allclose(box, qbox, atol=1e-12)
            self.assertAlmostEqual(abs(qm.box_chsh(box)), qm.TSIRELSON, places=12)
            self.assertLess(qm.signalling(box), 1e-12)

    # ------------------------------------------------------------ claim 7
    def test_claim7_multisimultaneity_flips_with_device_motion_m11_does_not(self):
        beta = 2500.0 / r.C_SI  # L-B2 device speed
        dx = 55.0                # L-B2 separation, metres (c = 1: times in metres)
        window = r.before_before_window(beta, dx)
        self.assertAlmostEqual(window / r.C_SI, 1.53e-12, delta=0.01e-12)  # 1.5 ps
        inside, outside = 0.5e-12 * r.C_SI, 3e-12 * r.C_SI
        # inside the window each device is first in its own rest frame (before-before)
        self.assertEqual(r.device_frame_first(0.0, 0.0, inside, dx, -beta, beta), ("A", "B"))
        # ... and reversing the device motions gives after-after
        self.assertEqual(r.device_frame_first(0.0, 0.0, inside, dx, beta, -beta), ("B", "A"))
        # outside the window both devices agree with the lab
        self.assertEqual(r.device_frame_first(0.0, 0.0, outside, dx, -beta, beta), ("A", "A"))
        # M11: the ordering is a function of the particles' (t, v) only; a
        # device velocity is not an input, and any global boost leaves it fixed
        for u in (-beta, beta, 0.5):
            first, _ = r.ordering_under_boost(self.g, u, "A", "B", TA, TB)
            self.assertEqual(first, "B")
        # undefined only on the measure-zero set tau_A = tau_B
        tB_eq = TA * np.sqrt(1 - VA**2) / np.sqrt(1 - VB**2)
        self.assertIsNone(LZ.order_in_pair_frame(self.g, "A", "B", TA, tB_eq))
        self.assertIsNotNone(LZ.order_in_pair_frame(self.g, "A", "B", TA, tB_eq * (1 + 1e-9)))


if __name__ == "__main__":
    unittest.main()
