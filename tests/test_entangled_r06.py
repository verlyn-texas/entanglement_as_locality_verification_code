import math
import unittest

import numpy as np

from mapping_spaces.entangled import qm, r06_qd_at_projection as r

Z, X = qm.direction(0.0), qm.direction(np.pi / 2)


class R06(unittest.TestCase):
    def test_claim1_qm_second_round_factorises(self):
        rho1 = r.after_first_projection(Z)
        self.assertAlmostEqual(qm.concurrence(rho1), 0.0, places=9)
        for outcome in (+1, -1):
            E2, mA, mB = r.second_round_qm(Z, X, X, outcome)
            self.assertAlmostEqual(E2, 0.0, places=12)
            self.assertAlmostEqual(mA, 0.0, places=12)
            self.assertAlmostEqual(mB, 0.0, places=12)
        # closed form (n1.n2)(-n1.n3) at generic settings, either outcome
        n1, n2, n3 = qm.direction(0.4, 0.2), qm.direction(1.1, -0.7), qm.direction(2.3, 1.9)
        for outcome in (+1, -1):
            E2, mA, mB = r.second_round_qm(n1, n2, n3, outcome)
            self.assertAlmostEqual(E2, r.closed_forms(n1, n2, n3)[0], places=12)
            self.assertAlmostEqual(E2, mA * mB, places=12)

    def test_claim2_influence_second_round(self):
        E2, mA, mB = r.second_round_influence(Z, X, X)
        self.assertAlmostEqual(E2, -1.0, places=12)
        self.assertAlmostEqual(mA, 0.0, places=12)
        self.assertAlmostEqual(mB, 0.0, places=12)
        n1, n2, n3 = qm.direction(0.4, 0.2), qm.direction(1.1, -0.7), qm.direction(2.3, 1.9)
        self.assertAlmostEqual(r.second_round_influence(n1, n2, n3)[0], r.closed_forms(n1, n2, n3)[1], places=12)
        # agreement when n2 || n1
        self.assertAlmostEqual(r.second_round_influence(n1, n1, n3)[0], r.second_round_qm(n1, n1, n3)[0], places=12)
        # grid: difference = n2_perp . n3_perp, max 1 at n2 = n3 perp n1
        thetas, E_qm, E_in = r.e2_grid(13)
        expected = -np.cos(thetas[:, None] - thetas[None, :]) + np.cos(thetas)[:, None] * np.cos(thetas)[None, :]
        np.testing.assert_allclose(E_in - E_qm, expected, atol=1e-12)
        dmax, t2, t3 = r.max_difference(13)
        self.assertAlmostEqual(dmax, 1.0, places=12)
        self.assertAlmostEqual(t2, np.pi / 2)
        self.assertAlmostEqual(t3, np.pi / 2)

    def test_claim3_influence_signals_and_violates_chsh(self):
        out = r.signalling_and_chsh()
        self.assertAlmostEqual(out["qm"]["signalling"], 0.0, places=12)
        self.assertAlmostEqual(out["influence"]["signalling"], 0.5, places=12)
        self.assertAlmostEqual(out["influence"]["P(b=-1|A=z,B=z)"], 1.0, places=12)
        self.assertAlmostEqual(out["influence"]["P(b=-1|A=x,B=z)"], 0.5, places=12)
        self.assertAlmostEqual(abs(out["qm"]["chsh"]), math.sqrt(2), places=12)
        self.assertLessEqual(abs(out["qm"]["chsh"]), 2.0)
        self.assertAlmostEqual(abs(out["influence"]["chsh"]), qm.TSIRELSON, places=12)
        # A's marginal never depends on B's setting in either box (A measures first)
        for model in ("qm", "influence"):
            box = r.second_round_box(model)
            pa = box.sum(axis=3)
            np.testing.assert_allclose(pa[:, 0, :], pa[:, 1, :], atol=1e-12)

    def test_claim4_qm_prediction_robust_to_preparation(self):
        for kind, param, E2 in r.robustness():
            self.assertAlmostEqual(E2, 0.0, places=12, msg=f"{kind} {param}")

    def test_claim5_trials(self):
        self.assertEqual(r.trials_needed(0.9994), 26)
        self.assertEqual(r.trials_needed(0.95), 39)
        self.assertEqual(r.trials_needed(0.75), 400)
        self.assertEqual(r.trials_needed(0.5), math.inf)
        self.assertAlmostEqual(r.observed_correlation(-1.0, 0.95), -0.81)
        for F in (0.9994, 0.95, 0.75):
            self.assertEqual(r.trials_needed(F), math.ceil(25 / (2 * F - 1) ** 4))

    def test_claim6_lab_tracks_identical_across_variants(self):
        kin = r.post_projection_kinematics(a=2.0, y=1.0)
        for name in ("M1a", "M1b", "M1c"):
            np.testing.assert_allclose(kin[name]["lab_tracks"], (-0.5, 0.5), atol=1e-12)
            self.assertAlmostEqual(kin[name]["sep_lab"], 1.0)
        self.assertAlmostEqual(kin["M1a"]["sep_frame_A"], 1.0)
        self.assertAlmostEqual(kin["M1a"]["sep_frame_B"], 1.0)
        self.assertAlmostEqual(kin["M1b"]["sep_frame_A"], 0.0)
        self.assertAlmostEqual(kin["M1b"]["sep_frame_B"], 1.0)
        self.assertAlmostEqual(kin["M1c"]["sep_frame_A"], 0.0)
        self.assertAlmostEqual(kin["M1c"]["sep_frame_B"], 0.0)

    def test_claim7_inert_variants_identical(self):
        st = r.variant_statistics()
        np.testing.assert_allclose(st["M1a"], st["M1b"], atol=1e-12)
        np.testing.assert_allclose(st["M1a"], st["M1c"], atol=1e-12)
        self.assertAlmostEqual(st["M1a"][0], 0.5 * (-np.cos(2.0)), places=12)
        self.assertAlmostEqual(st["M1c-influence"][0], -np.cos(2.0 - np.pi / 3), places=12)
        self.assertNotAlmostEqual(st["M1c-influence"][0], st["M1c"][0], places=3)


if __name__ == "__main__":
    unittest.main()
