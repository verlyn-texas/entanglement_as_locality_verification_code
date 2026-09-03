import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r05_continuous_qd as r


class R05(unittest.TestCase):
    a = 2.0  # lab separation rate 2/a = 1

    def test_claim1_at_rest_fraction_is_C_squared(self):
        # beta_12(z) = -2(1 - z^2)/a
        for z in (-1.0, -0.5, 0.0, 0.3, 1.0):
            self.assertAlmostEqual(frames.beta(frames.A, frames.B, self.a, z), -2 * (1 - z * z) / self.a)
        tab = r.locality_fraction_table(r.C_GRID, self.a)
        C = tab[:, 0]
        np.testing.assert_allclose(tab[:, 1], (1 - C * C) * 2 / self.a, atol=1e-12)   # rate in B
        np.testing.assert_allclose(tab[:, 3], C * C, atol=1e-12)                       # frames.locality_fraction
        np.testing.assert_allclose(tab[:, 4], C * C, atol=1e-12)                       # 1 - rate/(2/a)
        for Cv in r.C_GRID:                                                            # frame A agrees
            self.assertAlmostEqual(r.rates_at_rest(Cv, self.a)["A"], r.rates_at_rest(Cv, self.a)["B"])
        np.testing.assert_allclose(tab[:, 3], [0, 0.0625, 0.25, 0.5625, 1])

    def test_claim2_at_rest_lab_rate_is_not_2_over_a(self):
        tab = r.locality_fraction_table(r.C_GRID, self.a)
        C = tab[:, 0]
        np.testing.assert_allclose(tab[:, 2], 2 * (1 + C - C * C) / self.a, atol=1e-12)
        self.assertAlmostEqual(r.rates_at_rest(0.5, self.a)["lab"], 2.5 / self.a)
        self.assertAlmostEqual(r.rates_at_rest(0.0, self.a)["lab"], 2 / self.a)
        self.assertAlmostEqual(r.rates_at_rest(1.0, self.a)["lab"], 2 / self.a)
        # the piecewise-linear interpolant keeps the lab rate at 2/a for every C
        for Cv in r.C_GRID:
            self.assertAlmostEqual(r.rates_at_rest(Cv, self.a, "piecewise_linear")["lab"], 2 / self.a)

    def test_claim3_pinned_tracks_fraction_is_t_of_C_for_every_s(self):
        for Cv in r.C_GRID:
            for s_name in r.S_INTERP:
                p = r.rates_pinned(Cv, self.a, s_name, "linear")
                self.assertAlmostEqual(p["lab"], 2 / self.a)                    # tracks pinned
                self.assertAlmostEqual(p["B"], 2 * (1 - Cv) / self.a)           # s cancels
                self.assertAlmostEqual(p["A"], 2 * (1 - Cv) / self.a)
                pc = r.rates_pinned(Cv, self.a, s_name, "cubic")
                self.assertAlmostEqual(1 - pc["B"] / (2 / self.a), Cv ** 3)     # only t survives
            q = r.rates_pinned(Cv, self.a)                                       # quadratic own-frame paths
            self.assertAlmostEqual(q["pA"], Cv * (1 - Cv) / self.a)
            self.assertAlmostEqual(q["pB"], -Cv * (1 - Cv) / self.a)
        tab = r.interpolant_table(r.C_GRID, self.a)
        for s_name in r.S_INTERP:
            np.testing.assert_allclose(tab[(s_name, "linear", "pinned")], r.C_GRID, atol=1e-12)
        np.testing.assert_allclose(tab[("piecewise_linear", "linear", "at_rest")], r.C_GRID, atol=1e-12)
        np.testing.assert_allclose(tab[("quartic", "linear", "at_rest")], 2 * r.C_GRID ** 2 - r.C_GRID ** 4, atol=1e-12)
        self.assertEqual(r.repinned_fraction(0.3), 1.0)
        self.assertEqual(r.repinned_fraction(0.0), 0.0)

    def test_claim4_werner(self):
        tab = r.werner_table()
        np.testing.assert_allclose(tab[:, 1], tab[:, 2], atol=1e-9)     # C = max(0,(3p-1)/2)
        np.testing.assert_allclose(tab[:, 4], tab[:, 5], atol=1e-9)     # S = 2 sqrt2 p
        np.testing.assert_allclose(tab[:, 0], [1 / 3, 0.5, 1 / np.sqrt(2), 0.8, 0.9, 1.0])
        np.testing.assert_allclose(tab[:, 1], [0, 0.25, 0.5607, 0.7, 0.85, 1], atol=5e-5)
        np.testing.assert_allclose(tab[:, 3], [0, 0.0625, 0.3143, 0.49, 0.7225, 1], atol=5e-5)
        np.testing.assert_allclose(tab[:, 4], [0.9428, 1.4142, 2.0, 2.2627, 2.5456, 2.8284], atol=5e-5)

    def test_claim5_nonmax_pure(self):
        tab = r.nonmax_table()
        np.testing.assert_allclose(tab[:, 1], tab[:, 2], atol=1e-9)     # C = sin 2theta
        np.testing.assert_allclose(tab[:, 3], tab[:, 4], atol=1e-9)     # S = 2 sqrt(1 + C^2)
        np.testing.assert_allclose(tab[:, 3], [2.0, 2.1414, 2.4495, 2.7229, 2.8284], atol=5e-5)

    def test_claim6_C_is_not_a_function_of_S(self):
        self.assertAlmostEqual(r.werner_threshold_C(), 0.5607, places=4)
        d = r.same_C_different_S(0.7)
        self.assertAlmostEqual(d["C_werner"], 0.7, places=6)
        self.assertAlmostEqual(d["C_pure"], 0.7, places=6)
        self.assertAlmostEqual(d["S_werner"], 2.2627, places=4)
        self.assertAlmostEqual(d["S_pure"], 2.4413, places=4)
        e = r.same_S_different_C(0.8)
        self.assertAlmostEqual(e["S_werner"], e["S_pure"], places=9)
        self.assertAlmostEqual(e["C_werner"], 0.7, places=6)
        self.assertAlmostEqual(e["C_pure"], 0.5292, places=4)

    def test_claim7_ledger_illustration(self):
        L = r.ledger_illustration()
        v = L["L-A6d 0.1 K (verified)"]
        self.assertAlmostEqual(v["p"], 0.966, places=3)
        self.assertAlmostEqual(v["C"], 0.948, places=3)
        self.assertAlmostEqual(v["dC"], 0.047, places=3)
        self.assertAlmostEqual(v["C2"], 0.899, places=3)
        v = L["L-A6d 1.1 K (NOT in ledger; caller's brief)"]
        self.assertAlmostEqual(v["C"], 0.614, places=3)
        self.assertAlmostEqual(v["C2"], 0.377, places=3)
        v = L["L-A3 ions (verified)"]
        self.assertAlmostEqual(v["C"], 0.693, places=3)
        self.assertAlmostEqual(v["C2"], 0.481, places=3)
        self.assertAlmostEqual(v["F"], 0.847, places=3)   # vs the paper's ~0.88: not exactly Werner
        v = L["L-A4 photons (verified)"]
        self.assertAlmostEqual(v["C"], 1.000, places=3)
        self.assertLessEqual(v["p"], 1.0)                  # S never above 2 sqrt2

    def test_claim8_no_new_observable(self):
        out = r.m4_vs_qm(200, seed=5)
        self.assertEqual(out["max_box_difference"], 0.0)
        self.assertLess(out["max_signalling"], 1e-12)
        self.assertLess(out["C"].min(), 1e-9)              # states of every concurrence were sampled
        self.assertGreater(out["C"].max(), 0.99)
        self.assertLessEqual(out["S"].max(), qm.TSIRELSON + 1e-9)


if __name__ == "__main__":
    unittest.main()
