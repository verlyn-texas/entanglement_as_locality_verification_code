import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r01_baseline as r


class R01(unittest.TestCase):
    def test_claim1_singlet_chsh(self):
        out = r.chsh_all_frames()
        self.assertAlmostEqual(abs(out["S"]), qm.TSIRELSON, places=12)

    def test_claim2_no_signalling_marginals(self):
        np.testing.assert_allclose(r.marginals(), 0.5, atol=1e-12)

    def test_claim3_frame_invariance(self):
        out = r.chsh_all_frames()
        self.assertEqual(out["S_lab"], out["S_frame_A"])
        self.assertEqual(out["S_lab"], out["S_frame_B"])

    def test_claim4_colocated_in_particle_frames(self):
        out = r.chsh_all_frames(a_param=2.0, y_meas=1.0)
        self.assertAlmostEqual(out["sep_lab"], 1.0)
        self.assertAlmostEqual(out["sep_frame_A"], 0.0)
        self.assertAlmostEqual(out["sep_frame_B"], 0.0)

    def test_claim5_collapse(self):
        p, eB, C = r.collapse_example(0.7)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(eB, -1.0)
        self.assertAlmostEqual(C, 0.0, places=6)
        self.assertEqual(frames.lab_time_order(0.2, 0.9), frames.A)


if __name__ == "__main__":
    unittest.main()
