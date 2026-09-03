import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r10_cooper_pair_splitter as r10, r18_cps_proposal as r


class R18(unittest.TestCase):
    # ------------------------------------------------------------ claim 1: P1
    def test_claim1_polarisation_is_r10_fidelity(self):
        # <C> = -P1 P2 E  <=>  per-side fidelity F = (1+P)/2 in R10's channel
        for P1, P2 in ((0.6, 0.6), (0.98, 0.98), (0.9, 0.7)):
            self.assertAlmostEqual(r.s_obs_filters(1.0, 1.0, P1, P2), qm.TSIRELSON * P1 * P2, places=12)
            self.assertAlmostEqual(r.s_obs_filters(0.9, 0.8, P1, P2), qm.TSIRELSON * 0.9 * 0.8 * P1 * P2, places=12)
        self.assertAlmostEqual(r.fidelity_from_polarisation(0.6), 0.80)

    def test_claim1_platform1_numbers(self):
        p1 = r.platform1()
        self.assertAlmostEqual(p1["E1a_P1P2"], 0.36)
        self.assertAlmostEqual(p1["E1a_F_eff"], 0.80)
        self.assertAlmostEqual(p1["E1a_polarisation"], 1.018, places=3)
        self.assertAlmostEqual(p1["E1a_correlation"], 1.047, places=3)
        self.assertAlmostEqual(p1["E1b_visibility"], 2.546, places=3)
        self.assertAlmostEqual(p1["E2c"], 2.715, places=3)
        self.assertAlmostEqual(p1["E2c_times_eta"], 2.444, places=3)
        self.assertAlmostEqual(p1["E2c_P_sym"], 0.980, places=3)
        self.assertAlmostEqual(p1["E2c_F_eff"], 0.990, places=3)
        self.assertAlmostEqual(p1["P_min_eta1"], 0.8409, places=4)
        self.assertAlmostEqual(p1["P_min_eta085"], 0.9121, places=4)
        # the polarisation threshold is R10's F_min = 0.9204 in the other language
        self.assertAlmostEqual(r.fidelity_from_polarisation(p1["P_min_eta1"]), p1["F_min_r10"], places=12)
        self.assertAlmostEqual(p1["F_min_r10"], 0.9204, places=4)
        # exactly at threshold S = 2
        self.assertAlmostEqual(r.s_obs_filters(1.0, 1.0, p1["P_min_eta1"], p1["P_min_eta1"]), 2.0, places=12)
        self.assertAlmostEqual(r.s_obs_filters(1.0, 0.85, p1["P_min_eta085"], p1["P_min_eta085"]), 2.0, places=12)

    # ------------------------------------------------------------ claim 2: P2
    def test_claim2_platform2_table_and_thresholds(self):
        p2 = r.platform2()
        self.assertAlmostEqual(p2["p_W"], 0.9623, places=4)
        self.assertAlmostEqual(p2["S_ledger_dots_V"], 0.9656, places=4)
        np.testing.assert_allclose([p2["table_flip"][k] for k in (1.0, 0.99, 0.98, 0.95, 0.93, 0.90)],
                                   [2.678, 2.572, 2.468, 2.169, 1.981, 1.714], atol=6e-4)
        np.testing.assert_allclose([p2["table_depol"][k] for k in (1.0, 0.99, 0.98, 0.95, 0.93, 0.90)],
                                   [2.678, 2.625, 2.572, 2.417, 2.316, 2.169], atol=6e-4)
        self.assertAlmostEqual(p2["S_direct_fidelity"], 2.705, places=3)
        np.testing.assert_allclose([p2["Ft_min_flip"][e] for e in (1.0, 0.9, 0.85)], [0.9321, 0.9554, 0.9686], atol=6e-5)
        np.testing.assert_allclose([p2["Ft_min_depol"][e] for e in (1.0, 0.9, 0.85)], [0.8641, 0.9109, 0.9373], atol=6e-5)
        # thresholds are where S crosses 2; the flip model is the more pessimistic
        pW, Fr = p2["p_W"], r.LEDGER["F_readout_E3b"]
        for e in (1.0, 0.9, 0.85):
            self.assertAlmostEqual(r.s_obs_dots(pW, e, p2["Ft_min_flip"][e], Fr, "flip"), 2.0, places=12)
            self.assertAlmostEqual(r.s_obs_dots(pW, e, p2["Ft_min_depol"][e], Fr, "depol"), 2.0, places=12)
            self.assertGreater(p2["Ft_min_flip"][e], p2["Ft_min_depol"][e])
        # flip model == R10 closed form with the effective per-side fidelity
        Ft = 0.95
        F_eff = 0.5 * (1 + (2 * Ft - 1) * (2 * Fr - 1))
        self.assertAlmostEqual(r.s_obs_dots(pW, 1.0, Ft, Fr, "flip"), r10.s_obs(pW, 1.0, F_eff, F_eff), places=12)
        self.assertAlmostEqual(r.s_obs_dots(pW, 1.0, Ft, Fr, "flip"), r10.s_obs_direct(pW, 1.0, F_eff, F_eff), places=12)

    # ------------------------------------------------- claim 3: statistics
    def test_claim3_coincidences_and_run_time(self):
        p1, p2 = r.platform1(), r.platform2()
        t1 = r.run_time_table(p1["E2c"])
        self.assertEqual(t1["N"], 422)
        np.testing.assert_allclose([t1["t"][R] for R in r.RATES], [422, 42.2, 4.22, 0.422])
        t2 = r.run_time_table(p2["table_flip"][1.0])
        self.assertEqual(t2["N"], 480)
        t3 = r.run_time_table(p2["table_flip"][0.95])
        self.assertEqual(t3["N"], 9834)
        self.assertAlmostEqual(t3["t"][1.0] / 3600, 2.73, places=2)
        self.assertAlmostEqual(t3["t"][10.0] / 60, 16.4, places=1)
        self.assertEqual(r.run_time_table(p1["E1a_polarisation"])["N"], 0)
        self.assertTrue(np.isinf(r.run_time_table(p1["E1a_polarisation"])["t"][1.0]))
        self.assertAlmostEqual(r.electrons_per_second(100e-12) / 1e8, 6.24, places=2)
        self.assertAlmostEqual(r.max_cycle_rate(2.4e-6) / 1e5, 4.17, places=2)

    # ---------------------------------------------- claim 4: decisiveness
    def test_claim4_bands(self):
        b = r.decisiveness_bands(0.96, 10_000)
        self.assertTrue(b["predicted_violation"])
        self.assertAlmostEqual(b["V_threshold"], 0.7071, places=4)
        self.assertAlmostEqual(b["S_pred"], 2.715, places=3)
        self.assertAlmostEqual(b["sigma_S"], 0.0294, places=4)
        self.assertAlmostEqual(b["S_5sigma"], 2.147, places=3)
        np.testing.assert_allclose(b["band_pred"], [2.627, 2.803], atol=6e-4)
        self.assertFalse(r.decisiveness_bands(0.36, 10_000)["predicted_violation"])
        self.assertEqual(r.verdict_examples(), {"2DEG_S2.7": "consistent", "2DEG_S1.9": "mechanism fails",
                                                "2DEG_S2.1": "inconclusive", "nanowire_S1.0": "inconclusive"})

    # --------------------------------------------- claim 5: unequal arms
    def test_claim5_unequal_arms_timing_and_geometry(self):
        u = r.unequal_arms(v=1e4, LA=2e-6, ratios=(1.0, 2.0, 5.0))
        np.testing.assert_allclose([u[k]["yA"] for k in (1.0, 2.0, 5.0)], [0.2e-9] * 3, rtol=1e-12)
        np.testing.assert_allclose([u[k]["yB"] for k in (1.0, 2.0, 5.0)], [0.2e-9, 0.4e-9, 1.0e-9], rtol=1e-12)
        np.testing.assert_allclose([u[k]["dt"] for k in (1.0, 2.0, 5.0)], [0.0, 0.2e-9, 0.8e-9], atol=1e-22)
        self.assertEqual([u[k]["first"] for k in (1.0, 2.0, 5.0)], [0, frames.A, frames.A])
        for k in (1.0, 2.0, 5.0):
            o = u[k]["order_all_frames"]
            self.assertEqual(o["lab"], u[k]["first"])
            self.assertEqual(o["frame_A"], u[k]["first"])
            self.assertEqual(o["frame_B"], u[k]["first"])
            np.testing.assert_allclose(u[k]["frame_A_sep"], [0.0, 0.0], atol=1e-18)
            np.testing.assert_allclose(u[k]["frame_B_sep"], [0.0, 0.0], atol=1e-18)
        np.testing.assert_allclose([u[k]["lab_sep_at_yB"] for k in (1.0, 2.0, 5.0)], [4e-6, 8e-6, 20e-6], rtol=1e-12)
        self.assertLess(u[5.0]["dt_over_T_echo"], 1e-3)
        self.assertLess(u[5.0]["dt_over_t_readout"], 1e-3)
        self.assertAlmostEqual(u[5.0]["dt_over_T2star"], 0.08, places=12)
        # different drift speeds: v_B = 2 v_A flips which side is first
        w = r.unequal_arms(v=1e4, LA=2e-6, ratios=(1.0, 2.0, 5.0), vB=2e4)
        np.testing.assert_allclose([w[k]["dt"] for k in (1.0, 2.0, 5.0)], [-0.1e-9, 0.0, 0.3e-9], atol=1e-22)
        self.assertEqual([w[k]["first"] for k in (1.0, 2.0, 5.0)], [frames.B, 0, frames.A])
        for k in (1.0, 2.0, 5.0):
            np.testing.assert_allclose(w[k]["frame_A_sep"], [0.0, 0.0], atol=1e-18)

    def test_claim5_S_independent_of_ratio_and_null_sensitivity(self):
        for p, S_expect in ((1.0, qm.TSIRELSON), (0.96, 2.715)):
            s = r.s_vs_ratio(p)
            self.assertEqual([s[k]["first"] for k in (1.0, 2.0, 5.0)], [0, frames.A, frames.A])
            for k in (1.0, 2.0, 5.0):
                self.assertAlmostEqual(s[k]["S"], S_expect, places=3)
                self.assertAlmostEqual(s[k]["S"], s[1.0]["S"], places=12)
                self.assertLess(s[k]["max_box_diff"], 1e-12)
        n = r.null_test_sensitivity(r.platform1()["E2c"], 10_000)
        self.assertAlmostEqual(n["sigma_diff"], 0.0415, places=4)
        self.assertAlmostEqual(n["dS_5sigma"], 0.208, places=3)
        self.assertAlmostEqual(n["dS_2sigma"], 0.083, places=3)
        self.assertAlmostEqual(n["rel_5sigma"], 0.076, places=3)


if __name__ == "__main__":
    unittest.main()
