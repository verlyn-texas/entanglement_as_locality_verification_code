import unittest
import numpy as np
from mapping_spaces.entangled import qm, frames, r03_absolute_time as r


class R03(unittest.TestCase):
    def test_claim1_time_order_is_frame_invariant(self):
        orders = r.random_time_orders(n=200, seed=3)
        self.assertEqual(orders.shape, (200, 3))
        np.testing.assert_array_equal(orders[:, 0], orders[:, 1])
        np.testing.assert_array_equal(orders[:, 0], orders[:, 2])
        self.assertGreater((orders[:, 0] == frames.A).sum(), 0)
        self.assertGreater((orders[:, 0] == frames.B).sum(), 0)
        # y is literally untouched by the maps, for any height and frame pair
        rng = np.random.default_rng(1)
        y = rng.uniform(-5, 5, size=50)
        for k, l in ((3, 1), (3, 2), (1, 2), (2, 1)):
            _, yp, _ = frames.to_frame(k, l, 2.0, rng.uniform(-3, 3, 50), y, rng.uniform(-2, 2, 50))
            np.testing.assert_array_equal(yp, y)
        # equal times are simultaneous in every frame
        o = r.time_order_all_frames(1.3, 1.3, 2.0)
        self.assertEqual(set(o.values()), {0})

    def test_claim2_moving_lab(self):
        for v in (0.0, 0.5, -0.5, 50.0, -1e3):
            s = r.moving_lab_summary(2.0, v, y=1.0)
            self.assertAlmostEqual(s["sep_moving_lab"], 1.0)          # still 2y/a
            self.assertAlmostEqual(s["sep_lab"], 1.0)
            self.assertAlmostEqual(s["sep_frame_A"], 0.0)            # co-located
            self.assertAlmostEqual(s["sep_frame_B"], 0.0)
            self.assertAlmostEqual(s["xA_moving_lab"], -0.5 + v)     # tracks boosted by v
            self.assertAlmostEqual(s["xB_moving_lab"], 0.5 + v)
            self.assertAlmostEqual(s["beta_12_at_A"], 0.0)           # 1 <-> 2 untouched
            self.assertAlmostEqual(s["beta_12_at_B"], 0.0)
            self.assertLess(s["shift_defect"], 1e-9 * max(1.0, abs(v)))
            self.assertLess(s["closure_defect"], 1e-9 * max(1.0, abs(v)))
        # frame 3' at v = 0 is frame 3
        zz = np.linspace(-2, 2, 9)
        for k in (1, 2, 3):
            np.testing.assert_allclose(r.beta_v(k, r.MOVING_LAB, 2.0, zz, 0.0), frames.beta(k, 3, 2.0, zz))

    def test_claim3_chsh_independent_of_velocity_and_timing(self):
        S = r.chsh_scan(a=2.0, seed=7)
        self.assertEqual(S.size, 35)
        np.testing.assert_allclose(np.abs(S), qm.TSIRELSON, atol=1e-12)
        for v, yA, yB in ((0.0, 0.2, 0.9), (1e4 / 2.0, 0.9, 0.2), (-3.0, 1.0, 1.0)):
            out = r.chsh_moving_lab(v, yA, yB, a=2.0)
            self.assertAlmostEqual(abs(out["S"]), qm.TSIRELSON, places=12)
            self.assertLess(out["signalling"], 1e-12)
            self.assertEqual(out["order"], out["order_lab"])
            self.assertEqual(out["order"], frames.lab_time_order(yA, yB))
            # the sequential box is the QM box (projection order does not matter)
            a, ap, b, bp = qm.chsh_optimal_settings()
            np.testing.assert_allclose(out["box"], qm.box_from_state(qm.singlet(), (a, ap), (b, bp)), atol=1e-12)

    def test_claim4_influence_speed(self):
        self.assertEqual(r.influence_speed(1.0, 1.0, 2.0), float("inf"))
        self.assertAlmostEqual(r.influence_speed(1.0, 1.5, 2.0), 2.5)   # (1+1.5)/(2*0.5)
        self.assertAlmostEqual(r.influence_speed(2.0, 1.0, 2.0), 1.5)   # (2+1)/(2*1)
        # unbounded as the events approach simultaneity
        speeds = [r.influence_speed(1.0, 1.0 + d, 2.0) for d in (1e-1, 1e-3, 1e-6)]
        self.assertTrue(all(s2 > s1 for s1, s2 in zip(speeds, speeds[1:])))
        self.assertGreater(speeds[-1], 1e5)
        self.assertEqual(r.lab_event_separation(1.0, 1.0, 2.0), (1.0, 0.0))
        # in the particles' frames dx = 0: undefined
        for k in (frames.A, frames.B):
            self.assertTrue(np.isnan(r.influence_speed(1.0, 1.5, 2.0, k)))
            self.assertTrue(np.isnan(r.influence_speed(1.0, 1.0, 2.0, k)))

    def test_claim5_preferred_frame_bounds(self):
        b = r.ledger_bounds()
        self.assertAlmostEqual(b["L-B3"]["Vmin_over_c_u0"], 1 / 5.4e-6, places=6)   # 1.85e5
        self.assertAlmostEqual(b["L-B3"]["Vmin_over_c_u0"] / 1.8e5, 1.0, delta=0.03)  # ledger beta -> 0 figure
        self.assertAlmostEqual(b["L-B4"]["Vmin_over_c_u0"], 1.458e5, delta=1e3)
        self.assertAlmostEqual(b["L-B1"]["Vmin_over_c_u0"], 7.07e6, delta=1e4)
        # order of magnitude of every published bound (within a factor 11: orientation factors)
        for key in ("L-B3", "L-B4", "L-B1"):
            ratio = b[key]["Vmin_over_c_u0"] / b[key]["published_c"]
            self.assertTrue(1 / 11 < ratio < 11, (key, ratio))
            self.assertTrue(b[key]["absolute_time_persists"])
            self.assertFalse(b[key]["finite_below_bound_persists"])
            # a preferred-frame velocity beta <= 1e-3 shifts the bound by < 1e-2 relative
            self.assertAlmostEqual(b[key]["Vmin_over_c_umax"] / b[key]["Vmin_over_c_u0"], 1.0, delta=1e-2)
        # the decision rule itself
        self.assertTrue(r.correlation_persists(1.0, 0.0, float("inf")))
        self.assertFalse(r.correlation_persists(1.0, 0.0, 1e300))
        self.assertTrue(r.correlation_persists(1.0, 0.5, 2.0))
        self.assertFalse(r.correlation_persists(1.0, 0.5, 1.9))
        self.assertTrue(r.correlation_persists(1.0, 0.5, 1.9, u=0.5))    # frame moving with the influence
        self.assertEqual(r.minimum_speed(1.0, 0.0), float("inf"))
        self.assertAlmostEqual(r.minimum_speed(1.0, 0.5, u=0.5), 1.5)


if __name__ == "__main__":
    unittest.main()
