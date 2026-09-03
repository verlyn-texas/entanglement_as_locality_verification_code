import unittest
import numpy as np
from mapping_spaces.entangled import graph_frames as G, r15_qd_dimension as r


class R15(unittest.TestCase):
    def test_claim1_degrees(self):
        for n, d2 in ((2, 1), (3, 2), (5, 2), (10, 4), (20, 5), (50, 9)):
            self.assertEqual(r.degree_1d(n), n)
            self.assertEqual(r.degree_2d(n), d2)
            self.assertGreaterEqual((d2 + 1) * (d2 + 2) // 2, n + 1)
            self.assertLess(d2 * (d2 + 1) // 2, n + 1)

    def test_claim2_pinned_values_identical(self):
        g = r.example()
        pts, pots, worst = r.embed_2d(g)
        self.assertLess(worst, 1e-9)
        t1, t2 = r.pinned_table(g), r.pinned_table_2d(g, pts, pots)
        for k in t1:
            for j in t1[k]:
                self.assertAlmostEqual(t1[k][j], t2[k][j], places=8)
        self.assertTrue(g.is_local("A", "B") and g.is_local("C", "D"))

    def test_claim3_off_pinned_values_differ(self):
        g = r.example()
        pts, pots, _ = r.embed_2d(g)
        self.assertGreater(r.off_pinned_difference(g, pts, pots, "A", 0.3, 0.3), 1e-6)

    def test_claim4_same_z_allowed_in_2d(self):
        # two particles with the same z but different w and different velocities
        pts = [(0.0, 0.0), (1.0, 0.2), (1.0, -0.7)]
        vals = [0.0, 0.3, -0.5]
        f, res = r.bivariate_interpolant(pts, vals, d=1)
        self.assertLess(res, 1e-12)
        g = G.LocalityGraph().add(G.Particle("A", 1.0, 0.3))
        with self.assertRaises(ValueError):
            g.add(G.Particle("B", 1.0, -0.5))


if __name__ == "__main__":
    unittest.main()
