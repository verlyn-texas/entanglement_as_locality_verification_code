import unittest
import numpy as np
from mapping_spaces.entangled import qm, r02_contact_gate as r


class R02(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = r.summary()

    def test_claim1_deterministic_reach_4_but_signal(self):
        self.assertEqual(self.s["n_strategies"], 1024)
        self.assertAlmostEqual(self.s["max_abs_S"], 4.0)
        self.assertEqual(self.s["n_with_S4"], 128)
        self.assertTrue(self.s["all_S_above_2_signal"])

    def test_claim2_no_signalling_deterministic_are_local(self):
        self.assertEqual(self.s["n_no_signalling"], 256)
        self.assertAlmostEqual(self.s["max_abs_S_no_signalling"], 2.0)

    def test_claim3_pr_box_reachable(self):
        self.assertAlmostEqual(self.s["pr_S"], 4.0)
        self.assertLess(self.s["pr_signalling"], 1e-12)

    def test_claim4_qm_tuned_reproduces_singlet(self):
        self.assertAlmostEqual(abs(self.s["qm_tuned_S"]), qm.TSIRELSON, places=12)
        self.assertLess(self.s["qm_tuned_signalling"], 1e-12)
        a, ap, b, bp = qm.chsh_optimal_settings()
        np.testing.assert_allclose(r.qm_tuned_contact(), qm.box_from_state(qm.singlet(), (a, ap), (b, bp)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
