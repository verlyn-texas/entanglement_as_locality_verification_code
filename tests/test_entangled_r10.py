import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r10_cooper_pair_splitter as r


class R10(unittest.TestCase):
    def test_claim1_geometry_symmetric_arms(self):
        g = r.geometry(2.0, 0.8, 1.2)
        self.assertEqual(g["first"], frames.A)
        self.assertAlmostEqual(g["lab_at_yA"][0], -0.4)
        self.assertAlmostEqual(g["lab_at_yB"][1], 0.6)
        self.assertAlmostEqual(g["sep_lab_at_yA"], 0.8)
        self.assertAlmostEqual(g["sep_lab_at_yB"], 1.2)
        for fr in ("frame_A", "frame_B"):
            for t in ("at_yA", "at_yB"):
                self.assertAlmostEqual(g[f"sep_{fr}_{t}"], 0.0, places=12)

    def test_claim1_geometry_asymmetric_arms(self):
        g = r.asymmetric_arms(0.4, 0.6, 1.0, 0.9)
        self.assertAlmostEqual(g["a"], 2.0)
        self.assertEqual(g["first"], frames.B)  # y_B = 1.5 < y_A = 2.5
        np.testing.assert_allclose(g["model_lab_A"], g["phys_lab_A"], atol=1e-12)
        np.testing.assert_allclose(g["model_lab_B"], g["phys_lab_B"], atol=1e-12)
        for fr in ("frame_A", "frame_B"):
            for t in ("at_yA", "at_yB"):
                self.assertAlmostEqual(g[f"sep_{fr}_{t}"], 0.0, places=12)
        # in the pair frames the pair sits on the common drift (v_B - v_A) y / 2
        self.assertAlmostEqual(g["frame_A_at_yA"][0], 0.1 * 2.5)
        self.assertAlmostEqual(g["frame_B_at_yB"][1], 0.1 * 1.5)

    def test_claim2_closed_form_matches_direct_box(self):
        self.assertAlmostEqual(r.s_obs(1, 1, 1, 1), qm.TSIRELSON, places=12)
        self.assertAlmostEqual(r.s_obs(0.9, 0.8, 0.95, 0.97), r.s_obs_direct(0.9, 0.8, 0.95, 0.97), places=12)
        self.assertAlmostEqual(r.s_obs(0.9, 0.8, 0.95, 0.97), 1.7228515302, places=9)
        err, sig = r.s_obs_check_grid()
        self.assertLess(err, 1e-12)
        self.assertLess(sig, 1e-12)
        # each factor separately
        self.assertAlmostEqual(r.s_obs_direct(1, 1, 0.9, 1), qm.TSIRELSON * 0.8, places=12)
        self.assertAlmostEqual(r.s_obs_direct(1, 0.6, 1, 1), qm.TSIRELSON * 0.6, places=12)
        self.assertAlmostEqual(r.s_obs_direct(0.7, 1, 1, 1), qm.TSIRELSON * 0.7, places=12)

    def test_claim3_thresholds(self):
        fm, pm = r.threshold_tables()
        np.testing.assert_allclose([fm[k] for k in (1.0, 0.9, 0.8, 0.75)],
                                   [0.9204, 0.9432, 0.9701, 0.9855], atol=5e-5)
        np.testing.assert_allclose([pm[k] for k in (1.0, 0.95)], [0.7071, 0.8730], atol=5e-5)
        self.assertGreater(pm[0.92], 1.0)
        self.assertGreater(pm[0.90], 1.0)
        # the thresholds are exactly where S_obs crosses 2
        for pe in (1.0, 0.9, 0.8, 0.75):
            self.assertAlmostEqual(r.s_obs(pe, 1.0, fm[pe], fm[pe]), 2.0, places=12)
        self.assertAlmostEqual(r.s_obs(pm[0.95], 1.0, 0.95, 0.95), 2.0, places=12)

    def test_claim4_coincidences_for_5_sigma(self):
        self.assertEqual(r.n_5sigma_table(), {2.1: 28975, 2.3: 2975, 2.5: 975, 2.7: 445})
        self.assertEqual(r.n_5sigma(2.0), 0)
        for S in (2.3, 2.7):
            N = r.n_5sigma(S)
            self.assertGreaterEqual((S - 2) / r.sigma_S(S, N), 5.0 - 1e-9)
            mean, std = r.mc_sigma_S(S, N, trials=1500, seed=10)
            self.assertAlmostEqual(mean, S, delta=0.01)
            self.assertAlmostEqual(std / r.sigma_S(S, N), 1.0, delta=0.06)

    def test_claim5_order_absolute_and_irrelevant(self):
        self.assertEqual(r.order_all_frames(2.0, 0.8, 1.2), {"lab": frames.A, "frame_A": frames.A, "frame_B": frames.A})
        self.assertEqual(r.order_all_frames(0.5, 3.0, 1.0), {"lab": frames.B, "frame_A": frames.B, "frame_B": frames.B})
        for p in (1.0, 0.8):
            o = r.order_independence(p)
            self.assertLess(o["diff_AB"], 1e-12)
            self.assertLess(o["diff_BA"], 1e-12)
            self.assertAlmostEqual(o["S_AB"], o["S_ref"], places=12)
            self.assertAlmostEqual(o["S_BA"], o["S_ref"], places=12)
            self.assertAlmostEqual(o["S_ref"], qm.TSIRELSON * p, places=12)

    def test_claim6_benchmarks_and_verdict(self):
        b = r.benchmarks()
        self.assertAlmostEqual(b["V_A6d"], 0.9656, places=4)
        self.assertAlmostEqual(b["F_A6d"], 0.9913, places=4)
        self.assertAlmostEqual(b["V_A4"], 0.99970, places=5)
        self.assertEqual(r.verdict(2.4, 0.05, 0.95, 0.9, 0.97), "consistent")
        self.assertEqual(r.verdict(1.9, 0.05, 0.95, 0.9, 0.97), "mechanism fails")
        self.assertEqual(r.verdict(1.9, 0.05, 0.8, 0.8, 0.9), "inconclusive")
        self.assertEqual(r.verdict(2.1, 0.05, 0.95, 0.9, 0.97), "inconclusive")  # only 2 sigma


if __name__ == "__main__":
    unittest.main()
