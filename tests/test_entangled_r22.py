"""R22 — retarded graph updates and the chart-locality theorem (B5/C2)."""
import unittest

import numpy as np

from mapping_spaces.entangled import r16_covariant_swapping as r16
from mapping_spaces.entangled import r22_retarded_updates as r22


class TestRetardedReading(unittest.TestCase):
    def test_edge_existence_is_invariant(self):
        for ev, T, u in r22.random_configurations(400):
            self.assertTrue(r22.edge_existence_invariant(ev, T, (u, -u, 0.5)),
                            (ev, T, u))

    def test_cone_crossing_equivariant(self):
        rng = np.random.default_rng(9)
        for _ in range(200):
            v = rng.uniform(-0.9, 0.9)
            u = rng.uniform(-0.9, 0.9)
            x0, t0 = rng.uniform(-2, 2), rng.uniform(-2, 2)
            T = (t0 + rng.uniform(0.1, 2.0), rng.uniform(-2, 2))
            self.assertTrue(r22.cone_crossing_equivariant(v, x0, t0, T, u, tol=1e-7))

    def test_agrees_with_pair_clock_sign(self):
        # edge_exists(ev of measurement, T) iff the r16 pair clock kappa >= 0
        wl = r16.swap_worldlines()
        T = r16.transfer_event()
        for name, t_meas in r16.T_MEAS.items():
            w = wl[name]
            kappa = r16.pair_clock(w, T, t_meas)
            self.assertEqual(kappa >= 0, r22.edge_exists(w.event(t_meas), T))

    def test_hybrid_removed(self):
        # pre-revision hybrid: on the lab slice t = t_T + dt (small dt) the edge
        # existed everywhere; retarded: only inside the cone
        T = (2.0, 0.8)
        near = (2.1, 0.85)     # inside the cone (|dx| < dt)
        far = (2.1, 2.0)       # outside
        self.assertTrue(r22.edge_exists(near, T))
        self.assertFalse(r22.edge_exists(far, T))


class TestDelayedChoice(unittest.TestCase):
    def test_no_edge_at_outer_measurements_any_frame(self):
        out = r22.delayed_choice_retarded()
        for key, row in out.items():
            self.assertFalse(row["edge_at_measurement"], key)
            self.assertTrue(row["invariant"], key)


if __name__ == "__main__":
    unittest.main()
